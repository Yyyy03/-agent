from __future__ import annotations

FINRPT_OFFICIAL_RESPONSE_FIELDS = (
    "finance_write_response",
    "news_write_response",
    "report_write_response",
    "trend_write_response",
    "risk_response",
)

FINRPT_SOLVER_EXCLUDED_FIELDS = set(FINRPT_OFFICIAL_RESPONSE_FIELDS) | {
    "answer",
    "golden_answer",
    "ground_truth",
    "standard_answer",
    "reference",
    "label",
}

FINDOCRESEARCH_STRUCTURE = """Required FinDocResearch markdown structure:
# Section 1: Company Overview
## S1.1 Basic Information
## S1.2 Core Competencies
## S1.3 Mission & Vision

# Section 2: Financial Performance
## S2.1 Income Statement
## S2.2 Balance Sheet
## S2.3 Cash Flow Statement
## S2.4 Key Financial Metrics
## S2.5 Operating Performance

# Section 3: Business Analysis
## S3.1 Profitability Analysis
## S3.2 Financial Performance Summary
## S3.3 Business Competitiveness

# Section 4: Risk Factors
## S4.1 Risk Factors

# Section 5: Corporate Governance
## S5.1 Board Composition
## S5.2 Internal Controls

# Section 6: Future Outlook
## S6.1 Strategic Direction
## S6.2 Challenges and Uncertainties
## S6.3 Innovation and Development Plans
"""

FINDOCRESEARCH_FIELD_SCHEMA = """Official FinDocResearch field coverage:
- S1.1 Basic Information: Company Name, Establishment Date, Headquarters Location; category: Value.
- S1.2 Core Competencies: Innovation Advantages, Product Advantages, Brand Recognition, Reputation Ratings; categories: FY, FY-1.
- S1.3 Mission & Vision: Mission Statement, Vision Statement, Core Values; category: Value.
- S2.1 Income Statement: Revenue, Cost of Goods Sold, Gross Profit, Operating Expense, Operating Income, Net Profit, Income before Income Taxes, Income tax expense (benefit), Interest Expense; categories: FY, FY-1, FY-2.
- S2.2 Balance Sheet: Total Assets, Current Assets, Non-Current Assets, Total Liabilities, Current Liabilities, Non-Current Liabilities, Shareholders' Equity, Retained Earnings, Total Equity and Liabilities, Inventories, Prepaid Expenses; categories: FY, FY-1, FY-2.
- S2.3 Cash Flow Statement: Net Cash Flow from Operations, Net Cash Flow from Investing, Net Cash Flow from Financing, Net Increase/Decrease in Cash, Dividends; categories: FY, FY-1, FY-2.
- S2.4 Key Financial Metrics: Gross Margin, Operating Margin, Net Profit Margin, Current Ratio, Quick Ratio, Debt-to-Equity, Interest Coverage, Asset Turnover, Return on Equity, Return on Assets, Effective Tax Rate, Dividend Payout Ratio; categories: FY, FY-1, FY-2.
- S2.5 Operating Performance: Revenue by Product/Service, Revenue by Geographic Region; categories: FY, FY-1, FY-2.
- S3.1 Profitability Analysis: Revenue & Direct-Cost Dynamics, Operating Efficiency, External & One-Off Impact; category: Value.
- S3.2 Financial Performance Summary: Comprehensive Financial Health, Profitability and Earnings Quality, Operational Efficiency, Financial Risk Identification and Early Warning, Future Financial Performance Projection; categories: FY, FY-1.
- S3.3 Business Competitiveness: Business Model, Market Position; categories: FY, FY-1.
- S4.1 Risk Factors: Market Risks, Operational Risks, Financial Risks, Compliance Risks; categories: FY, FY-1.
- S5.1 Board Composition: Name, Position, Total Income; category: Board Member1.
- S5.2 Internal Controls: Risk Assessment Procedures, Control Activities, Monitoring Mechanisms, Identified Material Weaknesses or Deficiencies, Effectiveness; categories: FY, FY-1.
- S6.1 Strategic Direction: Mergers and Acquisition, New Technologies, Organisational Restructuring; categories: FY, FY-1.
- S6.2 Challenges and Uncertainties: Economic Challenges, Competitive Pressures; categories: FY, FY-1.
- S6.3 Innovation and Development Plans: R&D Investments, New Product Launches; categories: FY, FY-1.
"""

FINDOCRESEARCH_REQUIRED_SECTIONS = [
    "Section 1: Company Overview",
    "Section 2: Financial Performance",
    "Section 3: Business Analysis",
    "Section 4: Risk Factors",
    "Section 5: Corporate Governance",
    "Section 6: Future Outlook",
]

FINDOCRESEARCH_REQUIRED_SUBSECTIONS = [
    "S1.1", "S1.2", "S1.3",
    "S2.1", "S2.2", "S2.3", "S2.4", "S2.5",
    "S3.1", "S3.2", "S3.3",
    "S4.1",
    "S5.1", "S5.2",
    "S6.1", "S6.2", "S6.3",
]
