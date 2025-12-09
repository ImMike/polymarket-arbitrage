#!/usr/bin/env python3
"""
Run Trading Bot with Dashboard
===============================

Starts the trading bot and web dashboard together.
Supports cross-platform arbitrage between Polymarket and Kalshi.

Usage:
    python run_with_dashboard.py              # Dry run mode
    python run_with_dashboard.py --live       # Live mode
    python run_with_dashboard.py --port 8080  # Custom port
"""

import argparse
import asyncio
import logging
import signal
import sys
import threading
from datetime import datetime

import uvicorn

from polymarket_client import PolymarketClient
from kalshi_client import KalshiClient
from core.data_feed import DataFeed
from core.arb_engine import ArbEngine, ArbConfig
from core.execution import ExecutionEngine, ExecutionConfig
from core.risk_manager import RiskManager, RiskConfig
from core.portfolio import Portfolio
from core.cross_platform_arb import CrossPlatformArbEngine, MarketMatcher
from utils.config_loader import load_config, BotConfig
from utils.logging_utils import setup_logging
from dashboard.server import app, dashboard_state
from dashboard.integration import DashboardIntegration


logger = logging.getLogger(__name__)


class TradingBotWithDashboard:
    """Trading bot with integrated dashboard."""
    
    def __init__(self, config: BotConfig, port: int = 8888):
        self.config = config
        self.port = port
        self._running = False
        
        # Components - Polymarket
        self.client = None
        self.data_feed = None
        self.arb_engine = None
        self.execution_engine = None
        self.risk_manager = None
        self.portfolio = None
        self.dashboard_integration = None
        
        # Components - Kalshi (cross-platform)
        self.kalshi_client = None
        self.cross_platform_engine = None
        self.market_matcher = None
        self._kalshi_markets = []
        self._matched_pairs = []
        
        # Server
        self._server = None
        self._server_task = None
    
    async def start(self) -> None:
        """Start the bot and dashboard."""
        logger.info("=" * 60)
        logger.info("Polymarket + Kalshi Arbitrage Bot")
        logger.info("=" * 60)
        logger.info(f"Mode: {'DRY RUN' if self.config.is_dry_run else 'LIVE'}")
        logger.info(f"Cross-Platform: {'ENABLED' if self.config.mode.cross_platform_enabled else 'DISABLED'}")
        logger.info(f"Dashboard: http://localhost:{self.port}")
        logger.info("=" * 60)
        
        self._running = True
        
        # Initialize Polymarket API client
        self.client = PolymarketClient(
            rest_url=self.config.api.polymarket_rest_url,
            ws_url=self.config.api.polymarket_ws_url,
            gamma_url=self.config.api.gamma_api_url,
            api_key=self.config.api.api_key,
            private_key=self.config.api.private_key,
            timeout=self.config.api.timeout_seconds,
            dry_run=self.config.is_dry_run,
        )
        await self.client.connect()
        
        # Initialize Kalshi client (if cross-platform enabled)
        if self.config.mode.cross_platform_enabled and self.config.mode.kalshi_enabled:
            logger.info("Initializing Kalshi client for cross-platform arbitrage...")
            self.kalshi_client = KalshiClient(
                timeout=self.config.api.timeout_seconds,
                max_retries=self.config.api.max_retries,
                dry_run=self.config.is_dry_run,
            )
            
            # Initialize cross-platform arbitrage engine
            self.cross_platform_engine = CrossPlatformArbEngine(
                min_edge=self.config.trading.min_edge,
            )
            self.market_matcher = self.cross_platform_engine.matcher
            
            # Start Kalshi monitoring in background
            asyncio.create_task(self._start_kalshi_monitoring())
        
        # Initialize portfolio
        initial_balance = (
            self.config.mode.dry_run_initial_balance 
            if self.config.is_dry_run 
            else 0.0
        )
        self.portfolio = Portfolio(initial_balance=initial_balance)
        
        # Initialize risk manager
        self.risk_manager = RiskManager(RiskConfig(
            max_position_per_market=self.config.risk.max_position_per_market,
            max_global_exposure=self.config.risk.max_global_exposure,
            max_daily_loss=self.config.risk.max_daily_loss,
            max_drawdown_pct=self.config.risk.max_drawdown_pct,
            trade_only_high_volume=self.config.risk.trade_only_high_volume,
            min_24h_volume=self.config.risk.min_24h_volume,
            whitelist=self.config.risk.whitelist,
            blacklist=self.config.risk.blacklist,
            kill_switch_enabled=self.config.risk.kill_switch_enabled,
        ))
        
        # Initialize execution engine
        self.execution_engine = ExecutionEngine(
            client=self.client,
            risk_manager=self.risk_manager,
            portfolio=self.portfolio,
            config=ExecutionConfig(
                slippage_tolerance=self.config.trading.slippage_tolerance,
                order_timeout_seconds=self.config.trading.order_timeout_seconds,
                dry_run=self.config.is_dry_run,
            ),
        )
        await self.execution_engine.start()
        
        # Initialize arb engine
        self.arb_engine = ArbEngine(ArbConfig(
            min_edge=self.config.trading.min_edge,
            bundle_arb_enabled=self.config.trading.bundle_arb_enabled,
            min_spread=self.config.trading.min_spread,
            mm_enabled=self.config.trading.mm_enabled,
            tick_size=self.config.trading.tick_size,
            default_order_size=self.config.trading.default_order_size,
            min_order_size=self.config.trading.min_order_size,
            max_order_size=self.config.trading.max_order_size,
        ))
        
        # Initialize data feed
        market_ids = self.config.trading.markets.copy()
        self.data_feed = DataFeed(
            client=self.client,
            market_ids=market_ids,
            position_refresh_interval=5.0,
            on_update=self._on_market_update,
            config=self.config,
        )
        await self.data_feed.start()
        
        # Initialize dashboard integration
        self.dashboard_integration = DashboardIntegration(
            data_feed=self.data_feed,
            arb_engine=self.arb_engine,
            execution_engine=self.execution_engine,
            risk_manager=self.risk_manager,
            portfolio=self.portfolio,
            mode="dry_run" if self.config.is_dry_run else "live",
        )
        await self.dashboard_integration.start()
        
        # Start fill simulation for dry run
        if self.config.is_dry_run and self.config.mode.simulate_fills:
            asyncio.create_task(self._simulate_fills())
        
        # Start the web server
        await self._start_server()
        
        logger.info("Bot and dashboard started successfully!")
        logger.info(f"Open http://localhost:{self.port} in your browser")
    
    async def _start_server(self) -> None:
        """Start the uvicorn server."""
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=self.port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(self._server.serve())
    
    def _on_market_update(self, market_id: str, market_state) -> None:
        """Handle market updates."""
        if not self._running:
            return
        
        # Check risk limits
        if not self.risk_manager.within_global_limits():
            return
        
        # Analyze for opportunities
        signals = self.arb_engine.analyze(market_state)
        
        for signal in signals:
            # Add to dashboard
            if signal.opportunity:
                self.dashboard_integration.add_opportunity(
                    opportunity_type=signal.opportunity.opportunity_type.value,
                    market_id=signal.market_id,
                    edge=signal.opportunity.edge,
                    suggested_size=signal.opportunity.suggested_size,
                )
            
            self.dashboard_integration.add_signal(
                action=signal.action,
                market_id=signal.market_id,
            )
            
            # Submit to execution
            asyncio.create_task(self.execution_engine.submit_signal(signal))
    
    async def _simulate_fills(self) -> None:
        """Simulate order fills in dry run mode."""
        import random
        
        while self._running:
            try:
                await asyncio.sleep(2.0)
                
                orders = self.execution_engine.get_open_orders()
                for order in orders:
                    if random.random() < self.config.mode.fill_probability:
                        trade = self.client.simulate_fill(order.order_id)
                        if trade:
                            self.execution_engine.handle_fill(trade)
                            self.dashboard_integration.add_trade(
                                side=trade.side.value,
                                price=trade.price,
                                size=trade.size,
                                market_id=trade.market_id,
                            )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Fill simulation error: {e}")
    
    async def _start_kalshi_monitoring(self) -> None:
        """Start monitoring Kalshi markets for cross-platform arbitrage."""
        if not self.kalshi_client:
            return
        
        logger.info("Starting Kalshi market monitoring...")
        
        async with self.kalshi_client:
            # Set up dashboard for loading state
            dashboard_state.cross_platform["enabled"] = True
            dashboard_state.cross_platform["matching_status"] = "loading"
            
            # Fetch Kalshi markets with progress updates
            logger.info("Fetching Kalshi markets...")
            
            def on_kalshi_progress(count):
                dashboard_state.cross_platform["kalshi_markets"] = count
            
            self._kalshi_markets = await self.kalshi_client.list_all_markets(
                status="open",
                max_markets=5000,
                on_progress=on_kalshi_progress,
            )
            logger.info(f"✓ Loaded {len(self._kalshi_markets)} Kalshi markets")
            
            # Update dashboard state
            dashboard_state.cross_platform["kalshi_markets"] = len(self._kalshi_markets)
            
            # Wait for at least SOME Polymarket markets to load (start matching quickly!)
            logger.info("Waiting for Polymarket markets...")
            for i in range(30):  # Wait up to 30 seconds
                await asyncio.sleep(1)
                poly_count = len(self.data_feed._markets) if self.data_feed else 0
                
                # Update dashboard with current loading progress
                dashboard_state.cross_platform["polymarket_markets"] = poly_count
                
                # Start matching as soon as we have some markets from both platforms
                if poly_count >= 50:
                    logger.info(f"Got {poly_count} Polymarket markets - starting matching!")
                    break
                    
                if i % 5 == 0:
                    logger.info(f"Polymarket: {poly_count} markets loaded...")
            
            # Match markets between platforms (run in background so dashboard stays responsive)
            if self.data_feed and self._kalshi_markets:
                polymarket_markets = list(self.data_feed._markets.values())
                logger.info(f"Starting background matching: {len(polymarket_markets)} Polymarket x {len(self._kalshi_markets)} Kalshi")
                
                # Set initial status
                dashboard_state.cross_platform["matching_status"] = "starting"
                
                # Start matching as a background task (dashboard will show progress)
                asyncio.create_task(self._run_matching_background(polymarket_markets))
    
    async def _run_matching_background(self, polymarket_markets: list) -> None:
        """Run market matching in a thread pool so dashboard stays fully responsive."""
        import concurrent.futures
        
        try:
            dashboard_state.cross_platform["matching_status"] = "matching"
            total = len(polymarket_markets) * len(self._kalshi_markets)
            dashboard_state.cross_platform["matching_total"] = total
            
            # Run matching in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            
            def run_matching_sync():
                """Synchronous matching that runs in thread."""
                import asyncio
                # Create new event loop for this thread
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                
                try:
                    def on_progress(checked, total, matches_found):
                        dashboard_state.cross_platform["matching_checked"] = checked
                        dashboard_state.cross_platform["matching_progress"] = int(checked / total * 100) if total > 0 else 0
                        dashboard_state.cross_platform["matched_pairs"] = matches_found
                        
                        # Update display data incrementally (show latest matches)
                        cached_pairs = self.market_matcher.get_cached_pairs()
                        if cached_pairs:
                            display_data = []
                            for pair in cached_pairs[-50:]:  # Show latest 50
                                display_data.append({
                                    "poly_question": pair.polymarket_question,
                                    "kalshi_title": pair.kalshi_title,
                                    "similarity": pair.similarity_score,
                                    "category": pair.category,
                                })
                            dashboard_state.cross_platform["matched_pairs_data"] = display_data
                    
                    result = new_loop.run_until_complete(
                        self.market_matcher.find_matches(
                            polymarket_markets,
                            self._kalshi_markets,
                            on_progress=on_progress,
                        )
                    )
                    return result
                finally:
                    new_loop.close()
            
            self._matched_pairs = await loop.run_in_executor(executor, run_matching_sync)
            
            dashboard_state.cross_platform["matching_status"] = "complete"
            dashboard_state.cross_platform["matching_progress"] = 100
            dashboard_state.cross_platform["matched_pairs"] = len(self._matched_pairs)
            
            logger.info(f"✓ Matching complete! Found {len(self._matched_pairs)} pairs")
            
            # Prepare matched pairs data for dashboard display
            matched_pairs_display = []
            for pair in self._matched_pairs[:50]:
                matched_pairs_display.append({
                    "poly_question": pair.polymarket_question,
                    "kalshi_title": pair.kalshi_title,
                    "similarity": pair.similarity_score,
                    "category": pair.category,
                })
            
            dashboard_state.cross_platform["matched_pairs_data"] = matched_pairs_display
            
        except Exception as e:
            logger.error(f"Matching error: {e}")
            import traceback
            traceback.print_exc()
            dashboard_state.cross_platform["matching_status"] = "error"
    
    async def stop(self) -> None:
        """Stop everything gracefully."""
        logger.info("Shutting down...")
        self._running = False
        
        if self.dashboard_integration:
            await self.dashboard_integration.stop()
        
        if self.data_feed:
            await self.data_feed.stop()
        
        if self.execution_engine:
            await self.execution_engine.stop()
        
        if self.client:
            await self.client.disconnect()
        
        # Kalshi client is closed via async context manager in _start_kalshi_monitoring
        
        if self._server:
            self._server.should_exit = True
        
        # Final summary
        if self.portfolio:
            summary = self.portfolio.get_summary()
            logger.info("=" * 60)
            logger.info("Final Summary")
            logger.info("=" * 60)
            logger.info(f"Total PnL: ${summary['pnl']['total_pnl']:.2f}")
            logger.info(f"Trades: {summary['total_trades']}")
            logger.info(f"Win Rate: {summary['win_rate']:.1%}")
        
        # Cross-platform summary
        if self.cross_platform_engine:
            cp_stats = self.cross_platform_engine.get_stats()
            logger.info(f"Cross-Platform Opportunities: {cp_stats['total_opportunities']}")
            logger.info(f"Matched Market Pairs: {cp_stats['matched_pairs']}")
        
        logger.info("Shutdown complete")
    
    async def run_forever(self) -> None:
        """Run until interrupted."""
        try:
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass


async def main_async(args: argparse.Namespace) -> None:
    """Async main function."""
    # Load config
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)
    
    # Override mode
    if args.live:
        config.mode.trading_mode = "live"
    elif args.dry_run:
        config.mode.trading_mode = "dry_run"
    
    # Create and run bot with dashboard
    bot = TradingBotWithDashboard(config, port=args.port)
    
    # Handle shutdown
    loop = asyncio.get_event_loop()
    shutdown_event = asyncio.Event()
    
    def signal_handler():
        logger.info("Shutdown signal received")
        shutdown_event.set()
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            pass
    
    try:
        await bot.start()
        
        # Wait for shutdown
        await shutdown_event.wait()
        
    except KeyboardInterrupt:
        pass
    finally:
        await bot.stop()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Polymarket Arbitrage Bot with Live Dashboard"
    )
    
    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="Config file path"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8888,
        help="Dashboard port (default: 8888)"
    )
    
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run in live mode"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Run in dry-run mode (default)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(console_level=log_level)
    
    # Run
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nShutdown complete.")


if __name__ == "__main__":
    main()

