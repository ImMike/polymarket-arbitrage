"""
Kalshi API Client
=================

Client for interacting with Kalshi prediction market exchange.
Supports public market data endpoints (no authentication required).

API Documentation: https://docs.kalshi.com/getting_started/quick_start_market_data
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, AsyncIterator
import httpx

from kalshi_client.models import (
    KalshiMarket,
    KalshiOrderBook,
    KalshiEvent,
    KalshiSeries,
)
from polymarket_client.models import PriceLevel, OrderBook

logger = logging.getLogger(__name__)


class KalshiClient:
    """
    Async client for Kalshi prediction market API.
    
    Note: Uses the elections subdomain which provides access to ALL markets,
    not just election-related ones.
    """
    
    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
    
    def __init__(
        self,
        timeout: float = 30.0,
        max_retries: int = 3,
        dry_run: bool = True,
    ):
        """
        Initialize Kalshi client.
        
        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            dry_run: If True, don't place real orders (read-only mode)
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.dry_run = dry_run
        self._client: Optional[httpx.AsyncClient] = None
        self._markets_cache: dict[str, KalshiMarket] = {}
        
    async def __aenter__(self) -> "KalshiClient":
        """Async context manager entry."""
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            headers={"Accept": "application/json"}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def _get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """
        Make a GET request to the Kalshi API.
        
        Args:
            endpoint: API endpoint (without base URL)
            params: Query parameters
            
        Returns:
            JSON response as dictionary
        """
        if not self._client:
            raise RuntimeError("Client not initialized. Use async with context manager.")
        
        url = f"{self.BASE_URL}{endpoint}"
        
        for attempt in range(self.max_retries):
            try:
                response = await self._client.get(url, params=params)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limited
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                    await asyncio.sleep(wait_time)
                elif e.response.status_code == 404:
                    logger.debug(f"Not found: {endpoint}")
                    return {}
                else:
                    logger.error(f"HTTP error {e.response.status_code}: {e}")
                    raise
            except httpx.RequestError as e:
                logger.warning(f"Request error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1)
                else:
                    raise
        
        return {}
    
    # =========================================================================
    # SERIES ENDPOINTS
    # =========================================================================
    
    async def get_series(self, series_ticker: str) -> Optional[KalshiSeries]:
        """
        Get information about a series.
        
        Args:
            series_ticker: Series ticker (e.g., "KXHIGHNY")
            
        Returns:
            KalshiSeries object or None if not found
        """
        data = await self._get(f"/series/{series_ticker}")
        if not data or "series" not in data:
            return None
        
        s = data["series"]
        return KalshiSeries(
            ticker=s.get("ticker", series_ticker),
            title=s.get("title", ""),
            frequency=s.get("frequency", ""),
            category=s.get("category", ""),
        )
    
    # =========================================================================
    # EVENTS ENDPOINTS
    # =========================================================================
    
    async def get_event(self, event_ticker: str) -> Optional[KalshiEvent]:
        """
        Get information about an event.
        
        Args:
            event_ticker: Event ticker (e.g., "KXHIGHNY-25DEC08")
            
        Returns:
            KalshiEvent object or None if not found
        """
        data = await self._get(f"/events/{event_ticker}")
        if not data or "event" not in data:
            return None
        
        e = data["event"]
        return KalshiEvent(
            event_ticker=e.get("ticker", event_ticker),
            series_ticker=e.get("series_ticker", ""),
            title=e.get("title", ""),
            category=e.get("category", ""),
        )
    
    # =========================================================================
    # MARKETS ENDPOINTS
    # =========================================================================
    
    async def list_markets(
        self,
        status: str = "open",
        series_ticker: Optional[str] = None,
        event_ticker: Optional[str] = None,
        limit: int = 1000,
        cursor: Optional[str] = None,
    ) -> tuple[list[KalshiMarket], Optional[str]]:
        """
        List markets with optional filters.
        
        Args:
            status: Market status filter (open, closed, settled)
            series_ticker: Filter by series
            event_ticker: Filter by event
            limit: Maximum markets to return (max 1000)
            cursor: Pagination cursor
            
        Returns:
            Tuple of (list of markets, next cursor or None)
        """
        params = {"status": status, "limit": min(limit, 1000)}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if cursor:
            params["cursor"] = cursor
        
        data = await self._get("/markets", params=params)
        if not data or "markets" not in data:
            return [], None
        
        markets = []
        for m in data["markets"]:
            market = self._parse_market(m)
            if market:
                markets.append(market)
                self._markets_cache[market.ticker] = market
        
        next_cursor = data.get("cursor")
        return markets, next_cursor
    
    async def list_all_markets(
        self,
        status: str = "open",
        max_markets: int = 10000,
        on_progress: callable = None,  # Callback for progress updates
    ) -> list[KalshiMarket]:
        """
        Fetch all markets with pagination.
        
        Args:
            status: Market status filter
            max_markets: Maximum total markets to fetch
            on_progress: Optional callback(loaded_count) for progress updates
            
        Returns:
            List of all markets
        """
        all_markets = []
        cursor = None
        
        while len(all_markets) < max_markets:
            markets, next_cursor = await self.list_markets(
                status=status,
                limit=1000,
                cursor=cursor,
            )
            
            if not markets:
                break
            
            all_markets.extend(markets)
            logger.info(f"Kalshi: {len(all_markets)} markets loaded...")
            
            # Report progress
            if on_progress:
                try:
                    on_progress(len(all_markets))
                except:
                    pass
            
            if not next_cursor:
                break
            cursor = next_cursor
            
            # Small delay to avoid rate limiting
            await asyncio.sleep(0.2)
        
        logger.info(f"Kalshi: {len(all_markets)} total markets loaded ✓")
        return all_markets[:max_markets]
    
    async def get_market(self, ticker: str) -> Optional[KalshiMarket]:
        """
        Get a specific market by ticker.
        
        Args:
            ticker: Market ticker
            
        Returns:
            KalshiMarket object or None if not found
        """
        # Check cache first
        if ticker in self._markets_cache:
            return self._markets_cache[ticker]
        
        data = await self._get(f"/markets/{ticker}")
        if not data or "market" not in data:
            return None
        
        market = self._parse_market(data["market"])
        if market:
            self._markets_cache[ticker] = market
        return market
    
    def _parse_market(self, data: dict) -> Optional[KalshiMarket]:
        """Parse market data from API response."""
        try:
            # Prices come in cents, convert to dollars
            yes_price = data.get("yes_price", 0) / 100.0 if data.get("yes_price") else 0.0
            no_price = data.get("no_price", 0) / 100.0 if data.get("no_price") else 0.0
            
            # If no_price not given, derive from yes_price
            if no_price == 0 and yes_price > 0:
                no_price = 1.0 - yes_price
            
            # Parse close time
            close_time = None
            if data.get("close_time"):
                try:
                    close_time = datetime.fromisoformat(data["close_time"].replace("Z", "+00:00"))
                except:
                    pass
            
            return KalshiMarket(
                ticker=data.get("ticker", ""),
                event_ticker=data.get("event_ticker", ""),
                series_ticker=data.get("series_ticker", ""),
                title=data.get("title", ""),
                subtitle=data.get("subtitle", ""),
                yes_price=yes_price,
                no_price=no_price,
                status=data.get("status", ""),
                result=data.get("result"),
                volume=data.get("volume", 0),
                open_interest=data.get("open_interest", 0),
                close_time=close_time,
                category=data.get("category", ""),
            )
        except Exception as e:
            logger.warning(f"Failed to parse Kalshi market: {e}")
            return None
    
    # =========================================================================
    # ORDERBOOK ENDPOINTS
    # =========================================================================
    
    async def get_orderbook(self, ticker: str) -> Optional[KalshiOrderBook]:
        """
        Get order book for a market.
        
        Args:
            ticker: Market ticker
            
        Returns:
            KalshiOrderBook object or None if not found
        """
        data = await self._get(f"/markets/{ticker}/orderbook")
        if not data or "orderbook" not in data:
            return None
        
        ob = data["orderbook"]
        
        # Parse YES bids (prices in cents)
        yes_bids = []
        for level in ob.get("yes", []):
            if len(level) >= 2:
                price_cents = level[0]
                quantity = level[1]
                yes_bids.append(PriceLevel(
                    price=price_cents / 100.0,  # Convert to dollars
                    size=float(quantity)
                ))
        
        # Parse NO bids (prices in cents)
        no_bids = []
        for level in ob.get("no", []):
            if len(level) >= 2:
                price_cents = level[0]
                quantity = level[1]
                no_bids.append(PriceLevel(
                    price=price_cents / 100.0,
                    size=float(quantity)
                ))
        
        # Sort bids descending (best/highest first)
        yes_bids.sort(key=lambda x: x.price, reverse=True)
        no_bids.sort(key=lambda x: x.price, reverse=True)
        
        return KalshiOrderBook(
            ticker=ticker,
            yes_bids=yes_bids,
            no_bids=no_bids,
            timestamp=datetime.utcnow(),
        )
    
    async def get_orderbook_unified(self, ticker: str) -> Optional[OrderBook]:
        """
        Get order book in unified format (compatible with Polymarket).
        
        Args:
            ticker: Market ticker
            
        Returns:
            OrderBook object or None if not found
        """
        kalshi_ob = await self.get_orderbook(ticker)
        if not kalshi_ob:
            return None
        return kalshi_ob.to_unified_orderbook()
    
    # =========================================================================
    # STREAMING (Polling-based for public API)
    # =========================================================================
    
    async def stream_orderbooks(
        self,
        tickers: list[str],
        batch_size: int = 100,
        rotation_delay: float = 2.0,
    ) -> AsyncIterator[tuple[str, OrderBook]]:
        """
        Stream order books for multiple markets using polling.
        
        Args:
            tickers: List of market tickers to stream
            batch_size: Number of markets to fetch per batch
            rotation_delay: Delay between batches in seconds
            
        Yields:
            Tuple of (ticker, OrderBook) for each update
        """
        logger.info(f"Starting Kalshi orderbook stream for {len(tickers)} markets")
        
        while True:
            for i in range(0, len(tickers), batch_size):
                batch = tickers[i:i + batch_size]
                logger.debug(f"Fetching Kalshi orderbooks {i+1}-{min(i+batch_size, len(tickers))} of {len(tickers)}")
                
                # Fetch orderbooks in parallel
                tasks = [self.get_orderbook_unified(ticker) for ticker in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for ticker, result in zip(batch, results):
                    if isinstance(result, Exception):
                        logger.debug(f"Failed to get Kalshi orderbook for {ticker}: {result}")
                        continue
                    if result:
                        yield (ticker, result)
                
                await asyncio.sleep(rotation_delay)
    
    # =========================================================================
    # CATEGORY/SEARCH HELPERS
    # =========================================================================
    
    async def get_markets_by_category(self, category: str) -> list[KalshiMarket]:
        """
        Get all open markets in a category.
        
        Common categories: elections, economics, crypto, tech, entertainment
        """
        # Kalshi API doesn't have a direct category filter, so we fetch all
        # and filter client-side
        all_markets = await self.list_all_markets(status="open")
        return [m for m in all_markets if m.category.lower() == category.lower()]
    
    async def search_markets(self, query: str) -> list[KalshiMarket]:
        """
        Search markets by title.
        
        Args:
            query: Search query string
            
        Returns:
            List of matching markets
        """
        all_markets = await self.list_all_markets(status="open")
        query_lower = query.lower()
        return [
            m for m in all_markets 
            if query_lower in m.title.lower() or query_lower in m.subtitle.lower()
        ]

