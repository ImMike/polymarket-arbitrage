"""
Cross-Platform Arbitrage Engine
===============================

Detects arbitrage opportunities between Polymarket and Kalshi prediction markets.

When the same prediction is priced differently on both platforms, we can:
- Buy YES on cheaper platform, sell YES on expensive platform
- Or buy NO on cheaper platform, sell NO on expensive platform
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from typing import Optional

from polymarket_client.models import Market, OrderBook, Opportunity, OpportunityType

logger = logging.getLogger(__name__)


@dataclass
class MarketPair:
    """A matched pair of markets on Polymarket and Kalshi."""
    polymarket_id: str
    kalshi_ticker: str
    polymarket_question: str
    kalshi_title: str
    similarity_score: float
    category: str = ""
    
    # Timestamps
    matched_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def pair_id(self) -> str:
        """Unique identifier for this pair."""
        return f"poly:{self.polymarket_id}|kalshi:{self.kalshi_ticker}"


@dataclass
class CrossPlatformOpportunity:
    """Arbitrage opportunity between Polymarket and Kalshi."""
    opportunity_id: str
    market_pair: MarketPair
    
    # Direction: which platform to buy/sell on
    buy_platform: str  # "polymarket" or "kalshi"
    sell_platform: str
    token: str  # "YES" or "NO"
    
    # Prices
    buy_price: float
    sell_price: float
    
    # Edge calculation
    gross_edge: float  # sell_price - buy_price
    net_edge: float    # After fees
    edge_pct: float    # As percentage
    
    # Sizing
    suggested_size: float = 0.0
    max_size: float = 0.0  # Limited by liquidity on both sides
    
    # Liquidity available
    buy_liquidity: float = 0.0
    sell_liquidity: float = 0.0
    
    # Metadata
    detected_at: datetime = field(default_factory=datetime.utcnow)
    
    def __str__(self) -> str:
        return (
            f"CrossPlatformArb: Buy {self.token} on {self.buy_platform} @ ${self.buy_price:.3f}, "
            f"Sell on {self.sell_platform} @ ${self.sell_price:.3f} | "
            f"Net Edge: {self.edge_pct:.2%}"
        )


class MarketMatcher:
    """
    Matches similar markets between Polymarket and Kalshi.
    
    Uses text similarity, keyword matching, and sports-specific logic
    to find markets that represent the same underlying prediction.
    """
    
    # Keywords to normalize/remove for matching
    NOISE_WORDS = {
        "will", "the", "a", "an", "be", "to", "in", "on", "by", "at",
        "what", "who", "which", "when", "is", "are", "was", "were",
        "market", "prediction", "bet", "odds", "win", "winner"
    }
    
    # NFL team name mappings (full name -> abbreviations and variants)
    NFL_TEAMS = {
        "arizona cardinals": ["cardinals", "arizona", "ari"],
        "atlanta falcons": ["falcons", "atlanta", "atl"],
        "baltimore ravens": ["ravens", "baltimore", "bal"],
        "buffalo bills": ["bills", "buffalo", "buf"],
        "carolina panthers": ["panthers", "carolina", "car"],
        "chicago bears": ["bears", "chicago", "chi"],
        "cincinnati bengals": ["bengals", "cincinnati", "cin"],
        "cleveland browns": ["browns", "cleveland", "cle"],
        "dallas cowboys": ["cowboys", "dallas", "dal"],
        "denver broncos": ["broncos", "denver", "den"],
        "detroit lions": ["lions", "detroit", "det"],
        "green bay packers": ["packers", "green bay", "gb"],
        "houston texans": ["texans", "houston", "hou"],
        "indianapolis colts": ["colts", "indianapolis", "ind"],
        "jacksonville jaguars": ["jaguars", "jacksonville", "jax"],
        "kansas city chiefs": ["chiefs", "kansas city", "kc"],
        "las vegas raiders": ["raiders", "las vegas", "lv"],
        "los angeles chargers": ["chargers", "la chargers", "lac"],
        "los angeles rams": ["rams", "la rams", "lar"],
        "miami dolphins": ["dolphins", "miami", "mia"],
        "minnesota vikings": ["vikings", "minnesota", "min"],
        "new england patriots": ["patriots", "new england", "ne"],
        "new orleans saints": ["saints", "new orleans", "no"],
        "new york giants": ["giants", "ny giants", "nyg"],
        "new york jets": ["jets", "ny jets", "nyj"],
        "philadelphia eagles": ["eagles", "philadelphia", "phi"],
        "pittsburgh steelers": ["steelers", "pittsburgh", "pit"],
        "san francisco 49ers": ["49ers", "san francisco", "sf"],
        "seattle seahawks": ["seahawks", "seattle", "sea"],
        "tampa bay buccaneers": ["buccaneers", "tampa bay", "tb"],
        "tennessee titans": ["titans", "tennessee", "ten"],
        "washington commanders": ["commanders", "washington", "was"],
    }
    
    # NBA teams
    NBA_TEAMS = {
        "boston celtics": ["celtics", "boston"],
        "brooklyn nets": ["nets", "brooklyn"],
        "new york knicks": ["knicks", "new york"],
        "philadelphia 76ers": ["76ers", "sixers", "philadelphia"],
        "toronto raptors": ["raptors", "toronto"],
        "chicago bulls": ["bulls", "chicago"],
        "cleveland cavaliers": ["cavaliers", "cavs", "cleveland"],
        "detroit pistons": ["pistons", "detroit"],
        "indiana pacers": ["pacers", "indiana"],
        "milwaukee bucks": ["bucks", "milwaukee"],
        "atlanta hawks": ["hawks", "atlanta"],
        "charlotte hornets": ["hornets", "charlotte"],
        "miami heat": ["heat", "miami"],
        "orlando magic": ["magic", "orlando"],
        "washington wizards": ["wizards", "washington"],
        "denver nuggets": ["nuggets", "denver"],
        "minnesota timberwolves": ["timberwolves", "wolves", "minnesota"],
        "oklahoma city thunder": ["thunder", "okc"],
        "portland trail blazers": ["blazers", "portland"],
        "utah jazz": ["jazz", "utah"],
        "golden state warriors": ["warriors", "golden state"],
        "los angeles clippers": ["clippers", "la clippers"],
        "los angeles lakers": ["lakers", "la lakers"],
        "phoenix suns": ["suns", "phoenix"],
        "sacramento kings": ["kings", "sacramento"],
        "dallas mavericks": ["mavericks", "mavs", "dallas"],
        "houston rockets": ["rockets", "houston"],
        "memphis grizzlies": ["grizzlies", "memphis"],
        "new orleans pelicans": ["pelicans", "new orleans"],
        "san antonio spurs": ["spurs", "san antonio"],
    }
    
    def __init__(self, min_similarity: float = 0.5):  # Higher threshold for quality
        """
        Initialize matcher.
        
        Args:
            min_similarity: Minimum similarity score (0-1) to consider a match
        """
        self.min_similarity = min_similarity
        self._matched_pairs: dict[str, MarketPair] = {}
        
        # Build reverse lookup for team names
        self._team_lookup = {}
        for full_name, variants in {**self.NFL_TEAMS, **self.NBA_TEAMS}.items():
            self._team_lookup[full_name] = full_name
            for variant in variants:
                self._team_lookup[variant.lower()] = full_name
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        words = [w for w in words if w not in self.NOISE_WORDS]
        return ' '.join(words)
    
    def extract_teams(self, text: str) -> list[str]:
        """Extract team names from text."""
        text_lower = text.lower()
        found_teams = []
        
        # Check for team names (longest match first)
        for team_key in sorted(self._team_lookup.keys(), key=len, reverse=True):
            if team_key in text_lower:
                canonical = self._team_lookup[team_key]
                if canonical not in found_teams:
                    found_teams.append(canonical)
                    # Remove from text to avoid double matches
                    text_lower = text_lower.replace(team_key, "")
        
        return found_teams
    
    def extract_key_entities(self, text: str) -> set[str]:
        """Extract key entities (names, numbers, dates) from text."""
        entities = set()
        
        # Numbers and percentages
        entities.update(re.findall(r'\d+(?:\.\d+)?%?', text))
        
        # Capitalized words (likely names/entities)
        entities.update(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text))
        
        # Political terms
        political_terms = ["trump", "biden", "republican", "democrat", "gop", "dnc", "harris", "desantis", "election", "president"]
        for term in political_terms:
            if term in text.lower():
                entities.add(term)
        
        # Crypto terms
        crypto_terms = ["bitcoin", "btc", "ethereum", "eth", "crypto", "solana", "sol"]
        for term in crypto_terms:
            if term in text.lower():
                entities.add(term)
        
        return entities
    
    def extract_date(self, text: str) -> Optional[str]:
        """
        Extract date from text for event matching.
        
        Returns normalized date string like "2024-12-08" or None.
        """
        text_lower = text.lower()
        
        # Month names
        months = {
            'jan': '01', 'january': '01', 'feb': '02', 'february': '02',
            'mar': '03', 'march': '03', 'apr': '04', 'april': '04',
            'may': '05', 'jun': '06', 'june': '06', 'jul': '07', 'july': '07',
            'aug': '08', 'august': '08', 'sep': '09', 'september': '09',
            'oct': '10', 'october': '10', 'nov': '11', 'november': '11',
            'dec': '12', 'december': '12'
        }
        
        # Pattern: "Sep 8", "September 8", "Sep 8, 2024"
        for month_name, month_num in months.items():
            pattern = rf'{month_name}\.?\s+(\d{{1,2}})(?:,?\s+(\d{{4}}))?'
            match = re.search(pattern, text_lower)
            if match:
                day = match.group(1).zfill(2)
                year = match.group(2) or '2024'  # Default to current year
                return f"{year}-{month_num}-{day}"
        
        # Pattern: "12/8/24", "12-8-2024"
        match = re.search(r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})', text)
        if match:
            month = match.group(1).zfill(2)
            day = match.group(2).zfill(2)
            year = match.group(3)
            if len(year) == 2:
                year = '20' + year
            return f"{year}-{month}-{day}"
        
        return None
    
    def dates_match(self, date1: Optional[str], date2: Optional[str]) -> bool:
        """Check if two dates are the same or within 1 day."""
        if not date1 or not date2:
            return True  # If no dates, don't penalize
        return date1 == date2
    
    def is_sports_match(self, text1: str, text2: str) -> tuple[bool, float]:
        """
        Check if two texts refer to the same sports matchup.
        
        Returns:
            (is_match, confidence_score)
        """
        teams1 = self.extract_teams(text1)
        teams2 = self.extract_teams(text2)
        
        if len(teams1) >= 2 and len(teams2) >= 2:
            # Check if same two teams
            teams1_set = set(teams1[:2])
            teams2_set = set(teams2[:2])
            
            if teams1_set == teams2_set:
                # Also check dates match
                date1 = self.extract_date(text1)
                date2 = self.extract_date(text2)
                
                if self.dates_match(date1, date2):
                    return True, 0.95  # Very high confidence for exact team + date match
                else:
                    return False, 0.3  # Same teams but different dates - likely different games
            
            # Check if at least one team matches (and dates match)
            overlap = teams1_set & teams2_set
            if len(overlap) >= 1:
                date1 = self.extract_date(text1)
                date2 = self.extract_date(text2)
                if self.dates_match(date1, date2):
                    return True, 0.7 + (0.2 * len(overlap) / 2)
        
        return False, 0.0
    
    def is_same_person_event(self, text1: str, text2: str) -> tuple[bool, float]:
        """
        Check if two texts refer to the same person-related prediction.
        
        Examples:
        - "Will Trump win?" / "Trump wins 2024" -> True, 0.85
        - "Trump approval rating" / "Trump job approval" -> True, 0.8
        """
        # Extract key person names
        person_patterns = [
            r'\b(trump|biden|harris|desantis|obama|pence)\b',
            r'\b(musk|zuckerberg|bezos|gates)\b',
            r'\b(powell|yellen)\b',  # Fed chairs
        ]
        
        persons1 = set()
        persons2 = set()
        
        for pattern in person_patterns:
            persons1.update(re.findall(pattern, text1.lower()))
            persons2.update(re.findall(pattern, text2.lower()))
        
        if persons1 and persons2 and persons1 & persons2:
            # Same person mentioned - check context similarity
            # Extract action/event words
            action_words1 = set(re.findall(r'\b(win|lose|approve|poll|elect|resign|indicted?|convicted?)\w*\b', text1.lower()))
            action_words2 = set(re.findall(r'\b(win|lose|approve|poll|elect|resign|indicted?|convicted?)\w*\b', text2.lower()))
            
            if action_words1 & action_words2:
                return True, 0.85  # Same person + same type of prediction
            else:
                return True, 0.6  # Same person, different prediction type
        
        return False, 0.0
    
    def calculate_similarity(
        self,
        polymarket_question: str,
        kalshi_title: str,
    ) -> float:
        """
        Calculate similarity score between two market questions.
        
        Uses multiple matching strategies:
        1. Sports team + date matching
        2. Person/politician matching
        3. Fuzzy text similarity
        4. Entity overlap
        
        Returns:
            Float between 0 and 1
        """
        # First check for sports matchup (highest priority)
        is_sports, sports_score = self.is_sports_match(polymarket_question, kalshi_title)
        if is_sports and sports_score > 0.7:
            return sports_score
        
        # Check for same person/event predictions
        is_person, person_score = self.is_same_person_event(polymarket_question, kalshi_title)
        if is_person and person_score > 0.7:
            return person_score
        
        # Normalize texts
        norm_poly = self.normalize_text(polymarket_question)
        norm_kalshi = self.normalize_text(kalshi_title)
        
        # Base text similarity using SequenceMatcher (fuzzy matching)
        text_sim = SequenceMatcher(None, norm_poly, norm_kalshi).ratio()
        
        # Entity overlap bonus
        poly_entities = self.extract_key_entities(polymarket_question)
        kalshi_entities = self.extract_key_entities(kalshi_title)
        
        if poly_entities and kalshi_entities:
            entity_overlap = len(poly_entities & kalshi_entities) / max(len(poly_entities), len(kalshi_entities))
            # Weighted combination
            combined_sim = 0.5 * text_sim + 0.5 * entity_overlap
        else:
            combined_sim = text_sim
        
        # Boost if both mention same sport type
        sport_keywords = ["nfl", "nba", "mlb", "nhl", "football", "basketball", "baseball", "hockey"]
        poly_sports = [s for s in sport_keywords if s in polymarket_question.lower()]
        kalshi_sports = [s for s in sport_keywords if s in kalshi_title.lower()]
        
        if poly_sports and kalshi_sports and set(poly_sports) & set(kalshi_sports):
            combined_sim = min(1.0, combined_sim + 0.15)
        
        # Boost for crypto predictions mentioning same coin
        crypto_keywords = ["bitcoin", "btc", "ethereum", "eth", "solana", "sol"]
        poly_crypto = [c for c in crypto_keywords if c in polymarket_question.lower()]
        kalshi_crypto = [c for c in crypto_keywords if c in kalshi_title.lower()]
        
        if poly_crypto and kalshi_crypto and set(poly_crypto) & set(kalshi_crypto):
            combined_sim = min(1.0, combined_sim + 0.2)
        
        return combined_sim
    
    def _categorize_market(self, text: str) -> str:
        """Detect category from market text. Order matters - check politics before sports!"""
        text_lower = text.lower()
        
        # Politics FIRST (to avoid "win the election" matching sports)
        if any(x in text_lower for x in ['trump', 'biden', 'harris', 'president', 'election', 
            'democrat', 'republican', 'congress', 'senate', 'governor', 'mayor', 'vote', 
            'nominee', 'primary', 'presidential', 'prime minister', 'parliament']):
            return 'politics'
        
        # Crypto
        if any(x in text_lower for x in ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'token',
            'solana', 'sol', 'blockchain', 'defi', 'nft', 'fdv', 'market cap']):
            return 'crypto'
        
        # Finance/Economics  
        if any(x in text_lower for x in ['fed', 'interest rate', 'inflation', 'gdp', 'recession',
            'stock', 'nasdaq', 'dow', 's&p', 'treasury', 'tariff', 'federal reserve']):
            return 'finance'
        
        # Sports (check AFTER politics)
        sports_keywords = ['nfl', 'nba', 'mlb', 'nhl', 'premier league', 'champions league', 
            'super bowl', 'playoff', 'la liga', 'soccer', ' fc', 'basketball team', 
            'football team', 'hockey', 'world cup', 'stanley cup']
        if any(x in text_lower for x in sports_keywords):
            return 'sports'
        
        # Check for team names
        if any(x in text_lower for x in self._team_lookup.keys()):
            return 'sports'
        
        # Entertainment
        if any(x in text_lower for x in ['oscar', 'grammy', 'emmy', 'movie', 'film', 'album',
            'artist', 'actor', 'actress', 'netflix', 'spotify', 'best picture']):
            return 'entertainment'
        
        # Tech
        if any(x in text_lower for x in ['ai ', 'openai', 'gpt', 'google', 'apple', 'microsoft',
            'tesla', 'spacex', 'nvidia']):
            return 'tech'
        
        return 'other'
    
    async def find_matches(
        self,
        polymarket_markets: list[Market],
        kalshi_markets: list,  # list[KalshiMarket]
        on_progress: callable = None,  # Callback for progress updates
    ) -> list[MarketPair]:
        """
        Find matching markets between platforms using category-based matching.
        
        Args:
            polymarket_markets: List of Polymarket markets
            kalshi_markets: List of Kalshi markets
            on_progress: Optional callback(checked, total, matches_found) for live updates
            
        Returns:
            List of matched market pairs
        """
        import asyncio
        
        matches = []
        
        # Get all active markets
        active_poly = [m for m in polymarket_markets if m.active]
        active_kalshi = [m for m in kalshi_markets if m.is_active]
        
        # Group by category for faster matching
        logger.info("Categorizing markets for faster matching...")
        
        poly_by_cat: dict[str, list] = {}
        for m in active_poly:
            cat = self._categorize_market(m.question)
            if cat not in poly_by_cat:
                poly_by_cat[cat] = []
            poly_by_cat[cat].append(m)
        
        kalshi_by_cat: dict[str, list] = {}
        for m in active_kalshi:
            cat = self._categorize_market(m.title)
            if cat not in kalshi_by_cat:
                kalshi_by_cat[cat] = []
            kalshi_by_cat[cat].append(m)
        
        # Log category breakdown
        logger.info("=== CATEGORY BREAKDOWN ===")
        for cat in set(list(poly_by_cat.keys()) + list(kalshi_by_cat.keys())):
            p_count = len(poly_by_cat.get(cat, []))
            k_count = len(kalshi_by_cat.get(cat, []))
            logger.info(f"  {cat}: Polymarket={p_count}, Kalshi={k_count}")
        
        # Calculate total comparisons (only within categories)
        total_comparisons = sum(
            len(poly_by_cat.get(cat, [])) * len(kalshi_by_cat.get(cat, []))
            for cat in set(list(poly_by_cat.keys()) + list(kalshi_by_cat.keys()))
        )
        
        logger.info(f"Total comparisons (category-based): {total_comparisons:,}")
        logger.info(f"(vs {len(active_poly) * len(active_kalshi):,} if matching all-to-all)")
        
        checked = 0
        
        # Match within each category (skip 'other' - too noisy)
        priority_categories = ['sports', 'politics', 'crypto', 'finance', 'entertainment', 'tech']
        
        for category in priority_categories:
            poly_markets = poly_by_cat.get(category, [])
            kalshi_markets_cat = kalshi_by_cat.get(category, [])
            
            if not poly_markets or not kalshi_markets_cat:
                continue
            
            logger.info(f"Matching {category}: {len(poly_markets)} x {len(kalshi_markets_cat)}")
            
            for poly_market in poly_markets:
                best_match = None
                best_score = 0.0
                
                for kalshi_market in kalshi_markets_cat:
                    score = self.calculate_similarity(
                        poly_market.question,
                        kalshi_market.title
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_match = kalshi_market
                    
                    checked += 1
                
                # Yield VERY frequently to keep event loop responsive
                if checked % 500 == 0:
                    await asyncio.sleep(0.01)  # Small sleep to let web requests through
                    pct = (checked / total_comparisons * 100) if total_comparisons > 0 else 0
                    
                    if checked % 5000 == 0:
                        logger.info(f"Progress: {checked:,}/{total_comparisons:,} ({pct:.1f}%) - {len(matches)} matches")
                    
                    if on_progress:
                        try:
                            on_progress(checked, total_comparisons, len(matches))
                        except:
                            pass
                
                # After checking all Kalshi markets for this Poly market
                if best_match and best_score >= self.min_similarity:
                    pair = MarketPair(
                        polymarket_id=poly_market.market_id,
                        kalshi_ticker=best_match.ticker,
                        polymarket_question=poly_market.question,
                        kalshi_title=best_match.title,
                        similarity_score=best_score,
                        category=category,
                    )
                    matches.append(pair)
                    self._matched_pairs[pair.pair_id] = pair
                    
                    logger.info(
                        f"MATCHED [{category}]: '{poly_market.question[:35]}...' <-> '{best_match.title[:35]}...' "
                        f"(score: {best_score:.2f})"
                    )
        
        logger.info(f"=== MATCHING COMPLETE: {len(matches)} pairs found ===")
        return matches
    
    def get_cached_pairs(self) -> list[MarketPair]:
        """Get all cached market pairs."""
        return list(self._matched_pairs.values())


class CrossPlatformArbEngine:
    """
    Detects arbitrage opportunities between Polymarket and Kalshi.
    
    Monitors matched market pairs and alerts when prices diverge enough
    to create profitable cross-platform arbitrage.
    """
    
    def __init__(
        self,
        min_edge: float = 0.02,  # 2% minimum edge
        polymarket_taker_fee: float = 0.015,  # 1.5%
        kalshi_taker_fee: float = 0.01,  # ~1% estimate
        gas_cost: float = 0.02,  # Gas cost per order
    ):
        """
        Initialize cross-platform arb engine.
        
        Args:
            min_edge: Minimum edge required (after fees) to signal
            polymarket_taker_fee: Polymarket taker fee rate
            kalshi_taker_fee: Kalshi taker fee rate
            gas_cost: Estimated gas cost per order
        """
        self.min_edge = min_edge
        self.polymarket_taker_fee = polymarket_taker_fee
        self.kalshi_taker_fee = kalshi_taker_fee
        self.gas_cost = gas_cost
        
        self.matcher = MarketMatcher()
        self._opportunities: list[CrossPlatformOpportunity] = []
        self._opportunity_count = 0
    
    def check_arbitrage(
        self,
        market_pair: MarketPair,
        polymarket_ob: OrderBook,
        kalshi_ob: OrderBook,
    ) -> Optional[CrossPlatformOpportunity]:
        """
        Check for arbitrage opportunity between a matched market pair.
        
        Args:
            market_pair: The matched market pair
            polymarket_ob: Polymarket order book
            kalshi_ob: Kalshi order book (in unified format)
            
        Returns:
            CrossPlatformOpportunity if found, None otherwise
        """
        # Get best prices from both platforms
        poly_yes_ask = polymarket_ob.best_ask_yes
        poly_yes_bid = polymarket_ob.best_bid_yes
        poly_no_ask = polymarket_ob.best_ask_no
        poly_no_bid = polymarket_ob.best_bid_no
        
        kalshi_yes_ask = kalshi_ob.best_ask_yes
        kalshi_yes_bid = kalshi_ob.best_bid_yes
        kalshi_no_ask = kalshi_ob.best_ask_no
        kalshi_no_bid = kalshi_ob.best_bid_no
        
        # Check for valid prices
        if not all([poly_yes_ask, poly_yes_bid, kalshi_yes_ask, kalshi_yes_bid]):
            return None
        
        best_opp = None
        best_net_edge = 0.0
        
        # Check all possible arbitrage directions:
        
        # 1. Buy YES on Polymarket, sell YES on Kalshi
        if poly_yes_ask and kalshi_yes_bid:
            gross = kalshi_yes_bid - poly_yes_ask
            fees = (poly_yes_ask * self.polymarket_taker_fee + 
                    kalshi_yes_bid * self.kalshi_taker_fee + 
                    self.gas_cost * 2)
            net = gross - fees
            if net > best_net_edge and net >= self.min_edge:
                best_net_edge = net
                best_opp = self._create_opportunity(
                    market_pair=market_pair,
                    buy_platform="polymarket",
                    sell_platform="kalshi",
                    token="YES",
                    buy_price=poly_yes_ask,
                    sell_price=kalshi_yes_bid,
                    gross_edge=gross,
                    net_edge=net,
                    buy_liquidity=polymarket_ob.yes.asks.best_size or 0,
                    sell_liquidity=kalshi_ob.yes.bids.best_size or 0,
                )
        
        # 2. Buy YES on Kalshi, sell YES on Polymarket
        if kalshi_yes_ask and poly_yes_bid:
            gross = poly_yes_bid - kalshi_yes_ask
            fees = (kalshi_yes_ask * self.kalshi_taker_fee + 
                    poly_yes_bid * self.polymarket_taker_fee + 
                    self.gas_cost * 2)
            net = gross - fees
            if net > best_net_edge and net >= self.min_edge:
                best_net_edge = net
                best_opp = self._create_opportunity(
                    market_pair=market_pair,
                    buy_platform="kalshi",
                    sell_platform="polymarket",
                    token="YES",
                    buy_price=kalshi_yes_ask,
                    sell_price=poly_yes_bid,
                    gross_edge=gross,
                    net_edge=net,
                    buy_liquidity=kalshi_ob.yes.asks.best_size or 0,
                    sell_liquidity=polymarket_ob.yes.bids.best_size or 0,
                )
        
        # 3. Buy NO on Polymarket, sell NO on Kalshi
        if poly_no_ask and kalshi_no_bid:
            gross = kalshi_no_bid - poly_no_ask
            fees = (poly_no_ask * self.polymarket_taker_fee + 
                    kalshi_no_bid * self.kalshi_taker_fee + 
                    self.gas_cost * 2)
            net = gross - fees
            if net > best_net_edge and net >= self.min_edge:
                best_net_edge = net
                best_opp = self._create_opportunity(
                    market_pair=market_pair,
                    buy_platform="polymarket",
                    sell_platform="kalshi",
                    token="NO",
                    buy_price=poly_no_ask,
                    sell_price=kalshi_no_bid,
                    gross_edge=gross,
                    net_edge=net,
                    buy_liquidity=polymarket_ob.no.asks.best_size or 0,
                    sell_liquidity=kalshi_ob.no.bids.best_size or 0,
                )
        
        # 4. Buy NO on Kalshi, sell NO on Polymarket
        if kalshi_no_ask and poly_no_bid:
            gross = poly_no_bid - kalshi_no_ask
            fees = (kalshi_no_ask * self.kalshi_taker_fee + 
                    poly_no_bid * self.polymarket_taker_fee + 
                    self.gas_cost * 2)
            net = gross - fees
            if net > best_net_edge and net >= self.min_edge:
                best_net_edge = net
                best_opp = self._create_opportunity(
                    market_pair=market_pair,
                    buy_platform="kalshi",
                    sell_platform="polymarket",
                    token="NO",
                    buy_price=kalshi_no_ask,
                    sell_price=poly_no_bid,
                    gross_edge=gross,
                    net_edge=net,
                    buy_liquidity=kalshi_ob.no.asks.best_size or 0,
                    sell_liquidity=polymarket_ob.no.bids.best_size or 0,
                )
        
        if best_opp:
            self._opportunities.append(best_opp)
            logger.info(f"🎯 CROSS-PLATFORM ARB: {best_opp}")
        
        return best_opp
    
    def _create_opportunity(
        self,
        market_pair: MarketPair,
        buy_platform: str,
        sell_platform: str,
        token: str,
        buy_price: float,
        sell_price: float,
        gross_edge: float,
        net_edge: float,
        buy_liquidity: float,
        sell_liquidity: float,
    ) -> CrossPlatformOpportunity:
        """Create a cross-platform opportunity object."""
        self._opportunity_count += 1
        
        # Calculate max size based on available liquidity
        max_size = min(buy_liquidity, sell_liquidity)
        
        # Suggested size: smaller of max_size or $100 for safety
        suggested_size = min(max_size, 100.0)
        
        return CrossPlatformOpportunity(
            opportunity_id=f"xplat_{self._opportunity_count}",
            market_pair=market_pair,
            buy_platform=buy_platform,
            sell_platform=sell_platform,
            token=token,
            buy_price=buy_price,
            sell_price=sell_price,
            gross_edge=gross_edge,
            net_edge=net_edge,
            edge_pct=net_edge / buy_price if buy_price > 0 else 0,
            suggested_size=suggested_size,
            max_size=max_size,
            buy_liquidity=buy_liquidity,
            sell_liquidity=sell_liquidity,
        )
    
    def get_recent_opportunities(self, limit: int = 50) -> list[CrossPlatformOpportunity]:
        """Get most recent cross-platform opportunities."""
        return self._opportunities[-limit:]
    
    def get_stats(self) -> dict:
        """Get cross-platform arbitrage statistics."""
        return {
            "total_opportunities": len(self._opportunities),
            "matched_pairs": len(self.matcher.get_cached_pairs()),
            "avg_edge": (
                sum(o.net_edge for o in self._opportunities) / len(self._opportunities)
                if self._opportunities else 0
            ),
        }

