from __future__ import annotations

from ...tools import ToolActionSpec
from ..akshare_tool import AkShareTool


MARKET_DATA_DESCRIPTION = (
        "Structured market and financial data via routed providers: Tushare for precise China "
        "A-share, index, fund, futures, options, bond-yield, macro, and interbank-rate tables, "
        "AkShare only as an explicit escape hatch for broader China-market tables not covered by typed actions, "
        "Yahoo Finance for global historical "
        "prices, FRED/ALFRED for US macro series, World Bank for "
        "country-level macro series, and explicit official-source connectors for OECD, IEA, Comtrade, "
        "LBMA, SAFE/NFRA attachment tables, and specific exchange contract histories. These official "
        "connectors return raw structured records/tables for evidence; they do not compute final answers. "
        "For US-listed companies, financial_statement returns standardized "
        "fundamentals from SEC XBRL companyfacts (Revenue, Net income, EPS, assets/liabilities, cash "
        "flows, capex, dividends, etc.) with explicit period and unit labels; pass a year via period "
        "(e.g. '2023'), statement_type=income|balance|cash, and/or indicator=<line item> (any us-gaap "
        "concept the company reports) to focus the results. These are HISTORICAL as-reported GAAP values "
        "only \u2014 NOT forward guidance, NOT non-GAAP/operational metrics (e.g. ARPU, take-rate), and NOT "
        "segment breakdowns; for those use sec_reader/web_reader. For exact "
        "share-price returns, use price_history rows with explicit dates and close prices, verify the "
        "requested trading dates, and use close rather than adjusted close unless the task says otherwise."
    )

MARKET_DATA_ACTIONS = {
        "capabilities": ToolActionSpec("Return a compact guide to market_data source families for a query and recommended typed actions; does not fetch data.", optional=["query", "provider", "asset_class"]),
        "quote_snapshot": ToolActionSpec("Fetch a quote snapshot through AkShare.", optional=["ticker", "symbol", "query", "market", "asset_class", "limit", "source"]),
        "price_history": ToolActionSpec("Fetch historical price data through Tushare/AkShare/Yahoo Finance. China A-share unadjusted daily OHLCV is routed to Tushare.", required=["ticker", "start_date", "end_date"], optional=["asset_class", "market", "adjust", "provider", "source", "limit"]),
        "equity_price_history": ToolActionSpec("Fetch precise unadjusted China A-share daily OHLCV rows through Tushare daily.", required=["ticker", "start_date", "end_date"], optional=["provider", "adjust", "limit", "fields"]),
        "equity_daily_basic": ToolActionSpec("Fetch China A-share daily valuation/basic rows through Tushare daily_basic.", required=["ticker"], optional=["provider", "start_date", "end_date", "trade_date", "limit", "fields"]),
        "equity_factor_history": ToolActionSpec("Fetch China A-share adjustment factors or stock factor rows through Tushare adj_factor/stk_factor.", required=["ticker", "start_date", "end_date"], optional=["provider", "kind", "trade_date", "limit", "fields"]),
        "equity_financials": ToolActionSpec("Fetch China A-share financial statement/indicator rows through Tushare income, balancesheet, cashflow, or fina_indicator; returns compact field semantics for common statement fields.", required=["ticker"], optional=["provider", "statement_type", "period", "start_date", "end_date", "ann_date", "report_type", "limit", "fields", "indicator", "metric", "query"]),
        "equity_dividend": ToolActionSpec("Fetch China A-share dividend rows through Tushare dividend; returns compact field semantics for common dividend fields.", required=["ticker"], optional=["provider", "period", "ann_date", "record_date", "ex_date", "limit", "fields", "indicator", "metric", "query"]),
        "equity_universe": ToolActionSpec("Fetch China A-share stock universe rows through Tushare stock_basic.", optional=["provider", "exchange", "market", "list_status", "query", "ticker", "name", "limit", "fields"]),
        "index_history": ToolActionSpec("Fetch China index daily OHLCV rows through namespace routing: Tushare for standard SH/SZ indexes and AkShare SW for .SI/Shenwan indexes. index_code may be one code or an array of codes for batch retrieval.", required=["index_code", "start_date", "end_date"], optional=["provider", "trade_date", "period", "limit", "fields", "query"]),
        "index_weight": ToolActionSpec("Fetch China index constituent weights through namespace routing: Tushare for standard SH/SZ indexes, AkShare SW for .SI/Shenwan, and AkShare CSIndex for CSI/Zhongzheng. index_code may be one code or an array of codes for batch retrieval.", required=["index_code"], optional=["provider", "start_date", "end_date", "trade_date", "limit", "fields", "query"]),
        "index_constituents": ToolActionSpec("Fetch China index constituents through namespace routing: AkShare SW for .SI/Shenwan, AkShare CSIndex for CSI/Zhongzheng, and Tushare index_weight for standard indexes. index_code may be one code or an array of codes for batch retrieval.", required=["index_code"], optional=["provider", "start_date", "end_date", "trade_date", "limit", "fields", "query"]),
        "index_basic": ToolActionSpec("Fetch China index catalog rows through namespace routing: Tushare index_basic, AkShare SW catalog, or AkShare CSIndex catalog.", optional=["provider", "market", "publisher", "category", "query", "index_code", "level", "limit", "fields"]),
        "index_catalog": ToolActionSpec("Alias of index_basic with namespace routing for standard China, Shenwan (.SI), and CSIndex/Zhongzheng index catalogs.", optional=["provider", "market", "publisher", "category", "query", "index_code", "level", "limit", "fields"]),
        "fund_history": ToolActionSpec("Fetch China fund/ETF daily OHLCV rows through Tushare fund_daily.", required=["fund_code", "start_date", "end_date"], optional=["provider", "trade_date", "limit", "fields"]),
        "fund_nav": ToolActionSpec("Fetch China fund NAV rows through Tushare fund_nav.", required=["fund_code"], optional=["provider", "start_date", "end_date", "market", "limit", "fields"]),
        "fund_portfolio": ToolActionSpec("Fetch China fund portfolio holding rows through Tushare fund_portfolio.", required=["fund_code"], optional=["provider", "period", "ann_date", "start_date", "end_date", "limit", "fields"]),
        "fund_basic": ToolActionSpec("Fetch China fund catalog rows through Tushare fund_basic.", optional=["provider", "market", "status", "query", "limit", "fields"]),
        "cn_futures_basic": ToolActionSpec("Fetch China futures contract catalog rows through Tushare fut_basic.", required=["exchange"], optional=["provider", "fut_type", "query", "limit", "fields"]),
        "cn_futures_mapping": ToolActionSpec("Fetch China futures main/continuous contract mapping rows through Tushare fut_mapping.", required=["contract"], optional=["provider", "exchange", "date", "trade_date", "limit", "fields"]),
        "cn_futures_daily": ToolActionSpec("Fetch China futures contract daily OHLC rows through Tushare fut_daily.", required=["contract", "start_date", "end_date"], optional=["provider", "exchange", "trade_date", "limit", "fields"]),
        "cn_futures_warehouse": ToolActionSpec("Fetch China futures warehouse receipt rows through Tushare fut_wsr.", required=["date"], optional=["provider", "symbol", "exchange", "query", "limit", "fields"]),
        "cn_futures_settle": ToolActionSpec("Fetch China futures settlement rows through Tushare fut_settle.", required=["contract"], optional=["provider", "exchange", "date", "trade_date", "start_date", "end_date", "limit", "fields"]),
        "cn_options_basic": ToolActionSpec("Fetch China options contract catalog rows through Tushare opt_basic.", required=["exchange"], optional=["provider", "opt_code", "call_put", "query", "limit", "fields"]),
        "cn_options_daily": ToolActionSpec("Fetch China options daily OHLC rows through Tushare opt_daily.", required=["contract", "start_date", "end_date"], optional=["provider", "exchange", "trade_date", "limit", "fields"]),
        "option_chain_metrics": ToolActionSpec("Build a normalized China commodity option-chain metric table for a trade date, including contract parsing, underlying settlement alignment, intrinsic value, time value, moneyness, ranking, and coverage diagnostics.", required=["exchange", "trade_date"], optional=["variety", "option_type", "metric", "price_basis", "top_k", "limit"]),
        "cn_bond_yield_curve": ToolActionSpec("Fetch ChinaBond yield curve rows through Tushare yc_cb.", optional=["provider", "date", "trade_date", "curve_type", "curve_name", "start_date", "end_date", "query", "limit", "fields"]),
        "cn_macro_series": ToolActionSpec("Fetch China macro rows through Tushare cn_gdp/cn_cpi/cn_ppi/cn_m/cn_pmi.", required=["series"], optional=["provider", "quarter", "month", "date", "start_date", "end_date", "limit", "fields"]),
        "shibor_series": ToolActionSpec("Fetch SHIBOR fixing or quote rows through Tushare shibor/shibor_quote.", optional=["provider", "kind", "date", "start_date", "end_date", "limit", "fields"]),
        "financial_statement": ToolActionSpec("Facade for company fundamentals: US tickers resolve to standardized SEC XBRL; China A-share tickers route to typed Tushare equity_financials; rough AkShare statement paths are rejected for standard A-share fields.", required=["ticker"], optional=["statement_type", "period", "date", "indicator", "metric", "query", "limit", "provider", "fields"]),
        "macro_series": ToolActionSpec("Fetch macro series through explicit structured providers. Auto-routing only covers clearly-resolved FRED, World Bank, or Yahoo market series; use explicit provider for AkShare/OECD/IEA/ALFRED or specialized actions for SAFE/NFRA/trade.", required=["series"], optional=["start_date", "end_date", "date", "year", "limit", "query", "country", "indicator", "provider", "source", "vintage_date", "realtime_start", "realtime_end", "dataflow", "dataset", "key", "product", "flow", "unit"]),
        "trade_series": ToolActionSpec("Fetch official trade rows through Comtrade or WITS. Returns raw rows such as qty/netWgt/primaryValue; it does not choose the final answer field.", required=["provider"], optional=["reporter", "reporter_code", "partner", "partner_code", "flow", "flow_code", "cmd_code", "commodity_code", "product", "period", "year", "classification", "type_code", "freq_code", "limit"]),
        "futures_contract_history": ToolActionSpec("Fetch OHLC rows for a specific futures contract symbol/date range. Does not fall back to continuous contracts.", required=["contract", "start_date", "end_date"], optional=["provider", "exchange", "commodity", "contract_month", "contract_year", "field", "limit"]),
        "metals_price": ToolActionSpec("Fetch official precious metal benchmark rows, currently LBMA gold/silver/platinum/palladium where public JSON is available.", required=["provider", "metal"], optional=["session", "date", "start_date", "end_date", "limit"]),
        "official_attachment_table": ToolActionSpec("Fetch and parse official SAFE/NFRA attachment tables or embedded HTML tables. Returns raw sheet rows with row indices and source metadata; it does not answer the question.", required=["provider"], optional=["url", "doc_id", "query", "limit", "header_rows"]),
        "source_catalog": ToolActionSpec("Official catalog lookup for provider fred/world_bank/iea/oecd/comtrade only; not a China/Tushare/AkShare catalog. Returns catalog rows only.", required=["provider"], optional=["country", "query", "kind", "catalog", "dataflow", "dataset", "series", "key", "limit"]),
        "market_table": ToolActionSpec("Explicit AkShare function escape hatch only when no typed market_data action/provider covers the source.", required=["function"], optional=list(AkShareTool.actions["call"].optional)),
    }

TUSHARE_TYPED_ACTIONS = {
        "equity_price_history",
        "equity_daily_basic",
        "equity_factor_history",
        "equity_financials",
        "equity_dividend",
        "equity_universe",
        "index_history",
        "index_weight",
        "index_constituents",
        "index_basic",
        "index_catalog",
        "fund_history",
        "fund_nav",
        "fund_portfolio",
        "fund_basic",
        "cn_futures_basic",
        "cn_futures_mapping",
        "cn_futures_daily",
        "cn_futures_warehouse",
        "cn_futures_settle",
        "cn_options_basic",
        "cn_options_daily",
        "cn_bond_yield_curve",
        "cn_macro_series",
        "shibor_series",
    }
