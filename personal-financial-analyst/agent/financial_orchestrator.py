"""Financial Optimization Orchestrator Agent.

This agent demonstrates the orchestrator-workers pattern using Claude Agent SDK.
It fetches financial data from MCP servers and coordinates specialized sub-agents
to provide comprehensive financial optimization recommendations.
"""

import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Any

from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, AgentDefinition, query, AssistantMessage, ResultMessage, TextBlock

# Helper function
def _load_prompt(filename: str) -> str:
    """Load prompt from prompts directory."""
    prompt_path = Path(__file__).parent / "prompts" / filename
    return prompt_path.read_text()

async def _auto_approve_all(
    tool_name: str,
    input_data: dict,
    context
):
    """Auto-approve all tools without prompting."""
    logger.debug(f"Auto-approving tool: {tool_name}")
    from claude_agent_sdk import PermissionResultAllow
    return PermissionResultAllow()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


DATA_DIR: Path = Path(__file__).parent.parent / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw_data"
AGENT_OUTPUTS_DIR: Path = DATA_DIR / "agent_outputs"


def _ensure_directories():
    """Ensure all required directories exist."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    AGENT_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _save_json(
    data: dict,
    filename: str
):
    """Save data to JSON file."""
    filepath = RAW_DATA_DIR / filename
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved data to {filepath}")


def _detect_subscriptions(
    bank_transactions: list[dict],
    credit_card_transactions: list[dict]
) -> list[dict]:
    """Detect subscription services from recurring transactions.

    TODO: Implement logic to:
    1. Filter transactions marked as recurring
    2. Identify subscription patterns (monthly charges)
    3. Categorize by service type
    4. Calculate total monthly subscription cost

    Args:
        bank_transactions: List of bank transaction dicts
        credit_card_transactions: List of credit card transaction dicts

    Returns:
        List of subscription dictionaries with service name, amount, frequency
    """
    subscriptions = []

    def _parse_date(date_str: str) -> datetime | None:
        """Try a few common date formats."""
        if not date_str:
            return None

        formats = [
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%Y/%m/%d",
            "%Y-%m-%d %H:%M:%S",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except (ValueError, TypeError):
                continue
        return None

    def _extract_service_name(txn: dict) -> str:
        """Extract a likely service/merchant name from transaction fields."""
        return (
            txn.get("merchant")
            or txn.get("description")
            or txn.get("name")
            or txn.get("payee")
            or "Unknown Service"
        )

    def _detect_frequency(dates: list[datetime]) -> str:
        """Estimate billing frequency from transaction dates."""
        if len(dates) < 2:
            return "monthly"

        dates = sorted(dates)
        day_gaps = [
            (dates[i] - dates[i - 1]).days
            for i in range(1, len(dates))
        ]

        if not day_gaps:
            return "monthly"

        avg_gap = sum(day_gaps) / len(day_gaps)

        if 25 <= avg_gap <= 35:
            return "monthly"
        if 6 <= avg_gap <= 8:
            return "weekly"
        if 12 <= avg_gap <= 16:
            return "biweekly"
        if 80 <= avg_gap <= 100:
            return "quarterly"
        if 350 <= avg_gap <= 380:
            return "yearly"

        return "monthly"

    def _normalize_amount(amount: Any) -> float | None:
        """Convert amount to float if possible."""
        try:
            return float(amount)
        except (TypeError, ValueError):
            return None

    all_transactions = bank_transactions + credit_card_transactions

    # Group candidate recurring outflows by service name
    grouped = defaultdict(list)

    for txn in all_transactions:
        recurring = txn.get("recurring", False)
        amount = _normalize_amount(txn.get("amount"))

        # Subscriptions are typically recurring negative outflows
        if not recurring or amount is None or amount >= 0:
            continue

        service = _extract_service_name(txn).strip()
        grouped[service].append(txn)

    subscriptions = []

    for service, txns in grouped.items():
        amounts = []
        dates = []

        for txn in txns:
            amt = _normalize_amount(txn.get("amount"))
            if amt is not None:
                amounts.append(abs(amt))

            parsed_date = _parse_date(txn.get("date"))
            if parsed_date:
                dates.append(parsed_date)

        if not amounts:
            continue

        # Use average absolute charge in case there are small tax/price differences
        avg_amount = round(sum(amounts) / len(amounts), 2)
        frequency = _detect_frequency(dates)

        subscriptions.append({
            "service": service,
            "amount": avg_amount,
            "frequency": frequency
        })

    return subscriptions


async def _fetch_financial_data(
    username: str,
    start_date: str,
    end_date: str
) -> tuple[dict, dict]:
    """Fetch data from Bank and Credit Card MCP servers.

    TODO: Implement MCP server connections using Claude Agent SDK
    1. Configure MCP server connections (ports 5001, 5002)
    2. Call get_bank_transactions tool
    3. Call get_credit_card_transactions tool
    4. Save raw data to files

    Args:
        username: Username for the account
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        Tuple of (bank_data, credit_card_data) dictionaries
    """
    logger.info(f"Fetching financial data for {username} from {start_date} to {end_date}")

    # TODO: Configure and connect to MCP servers
    # Example MCP configuration (keys must match FastMCP server names exactly):
    # mcp_servers = {
    #     "Bank Account Server": {  # Must match FastMCP("Bank Account Server")
    #         "type": "sse",
    #         "url": "http://127.0.0.1:5001"
    #     },
    #     "Credit Card Server": {  # Must match FastMCP("Credit Card Server")
    #         "type": "sse",
    #         "url": "http://127.0.0.1:5002"
    #     }
    # }

    mcp_servers = {
        "Bank Account Server": {
            "type": "http",
            "url": "http://127.0.0.1:5001/mcp"
        },
        "Credit Card Server": {
            "type": "http",
            "url": "http://127.0.0.1:5002/mcp"
        }
    }

    async def _run_tool_prompt(prompt: str) -> dict:
        """Run a prompt through Claude Agent SDK and parse final JSON response."""
        options = ClaudeAgentOptions(
            mcp_servers=mcp_servers,
            permission_mode="bypassPermissions"
        )

        final_text = ""

        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        final_text += block.text

        if not final_text:
            raise RuntimeError("No response returned from Claude Agent SDK")

        # Strip markdown code fences if Claude wrapped the response
        stripped = final_text.strip()
        if stripped.startswith("```"):
            stripped = stripped.split("\n", 1)[-1]
            stripped = stripped.rsplit("```", 1)[0].strip()

        try:
            return json.loads(stripped)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {final_text}")
            raise ValueError(f"Tool response was not valid JSON: {e}") from e

    bank_prompt = f"""
Use the MCP tool get_bank_transactions from the Bank Account Server.

Arguments:
- username: "{username}"
- start_date: "{start_date}"
- end_date: "{end_date}"

Return only the raw JSON response from the tool and nothing else.
""".strip()

    credit_card_prompt = f"""
Use the MCP tool get_credit_card_transactions from the Credit Card Server.

Arguments:
- username: "{username}"
- start_date: "{start_date}"
- end_date: "{end_date}"

Return only the raw JSON response from the tool and nothing else.
""".strip()

    bank_data = await _run_tool_prompt(bank_prompt)
    credit_card_data = await _run_tool_prompt(credit_card_prompt)

    _save_json(bank_data, "bank_transactions.json")
    _save_json(credit_card_data, "credit_card_transactions.json")

    return bank_data, credit_card_data


async def _run_orchestrator(
    username: str,
    start_date: str,
    end_date: str,
    user_query: str
):
    """Main orchestrator agent logic.

    TODO: Implement the orchestrator pattern:
    1. Fetch data from MCP servers (use tools)
    2. Perform initial analysis (detect subscriptions, anomalies)
    3. Decide which sub-agents to invoke based on query
    4. Define sub-agents using AgentDefinition
    5. Invoke sub-agents (can be parallel)
    6. Read and synthesize sub-agent results
    7. Generate final report

    Args:
        username: Username for the account
        start_date: Start date for analysis
        end_date: End date for analysis
        user_query: User's financial question/request
    """
    logger.info(f"Starting financial optimization orchestrator")
    logger.info(f"User query: {user_query}")

    _ensure_directories()

    # Step 1: Fetch financial data from MCP servers
    bank_data, credit_card_data = await _fetch_financial_data(
        username,
        start_date,
        end_date
    )

    # Step 2: Initial analysis
    logger.info("Performing initial analysis...")

    bank_transactions = bank_data.get("transactions", [])
    credit_card_transactions = credit_card_data.get("transactions", [])

    subscriptions = _detect_subscriptions(
        bank_transactions,
        credit_card_transactions
    )

    logger.info(f"Detected {len(subscriptions)} subscriptions")

    # Step 3: Define sub-agents
    # TODO: Define sub-agents using AgentDefinition
    # Example:
    # research_agent = AgentDefinition(
    #     description="Research cheaper alternatives for subscriptions and services",
    #     prompt="""You are a research specialist focused on finding cost savings.
    #     Your job is to research alternatives for subscriptions and services,
    #     compare features, pricing, and provide detailed recommendations.
    #     Write your findings to data/agent_outputs/research_results.json""",
    #     tools=["web_search", "write"],
    #     model="haiku"  # Fast and cheap for research
    # )

    # Step 3: Define sub-agents
    research_agent = AgentDefinition(
        description="Research cheaper alternatives for subscriptions and services",
        prompt=_load_prompt("research_agent_prompt.txt"),
        tools=["write"],
        model="haiku"
    )

    negotiation_agent = AgentDefinition(
        description="Create negotiation strategies and scripts for bills and services",
        prompt=_load_prompt("negotiation_agent_prompt.txt"),
        tools=["write"],
        model="haiku"
    )

    tax_agent = AgentDefinition(
        description="Identify tax-deductible expenses and optimization opportunities",
        prompt=_load_prompt("tax_agent_prompt.txt"),
        tools=["write"],
        model="haiku"
    )

    agents = {
        "research_agent": research_agent,
        "negotiation_agent": negotiation_agent,
        "tax_agent": tax_agent,
    }

    # agents = {
    #     # "research_agent": research_agent,
    #     # "negotiation_agent": negotiation_agent,
    #     # "tax_agent": tax_agent,
    # }

    # Step 4: Configure orchestrator agent with sub-agents
    # TODO: Create ClaudeAgentOptions with agents and MCP servers
    # options = ClaudeAgentOptions(
    #     model="sonnet",
    #     system_prompt="""You are a financial optimization coordinator.
    #     You have access to bank and credit card data.
    #     Analyze spending, delegate tasks to specialized agents, and synthesize
    #     their findings into actionable recommendations.""",
    #     agents=agents,
    #     # Add MCP server configurations here
    # )

    # Step 4: Configure orchestrator agent with sub-agents
    mcp_servers = {
        "Bank Account Server": {
            "type": "http",
            "url": "http://127.0.0.1:5001/mcp"
        },
        "Credit Card Server": {
            "type": "http",
            "url": "http://127.0.0.1:5002/mcp"
        }
    }

    working_dir = Path(__file__).parent.parent  # personal-financial-analyst/

    options = ClaudeAgentOptions(
        model="sonnet",
        system_prompt=_load_prompt("orchestrator_system_prompt.txt"),
        mcp_servers=mcp_servers,
        agents=agents,
        permission_mode="bypassPermissions",
        cwd=str(working_dir),
    )

    # Step 5: Run orchestrator with Claude Agent SDK
    # TODO: Use ClaudeSDKClient to run the orchestration
    # Example:
    # async with ClaudeSDKClient(options=options) as client:
    #     prompt = f"""Analyze my financial data and {user_query}
    #
    #     I have:
    #     - {len(bank_transactions)} bank transactions
    #     - {len(credit_card_transactions)} credit card transactions
    #     - {len(subscriptions)} identified subscriptions
    #
    #     Please:
    #     1. Identify opportunities for savings
    #     2. Delegate research to the research agent
    #     3. Delegate negotiation strategies to the negotiation agent
    #     4. Delegate tax analysis to the tax agent
    #     5. Read their results and create a final report
    #     """
    #
    #     async for message in client.stream(prompt):
    #         if message.type == "assistant":
    #             print(message.content)

    # Step 5: Run orchestrator with Claude Agent SDK
    prompt = f"""Analyze my financial data and {user_query}

    Account details:
    - Username: {username}
    - Date range: {start_date} to {end_date}

    I have already fetched and saved the raw data:
    - {len(bank_transactions)} bank transactions (saved to data/raw_data/bank_transactions.json)
    - {len(credit_card_transactions)} credit card transactions (saved to data/raw_data/credit_card_transactions.json)
    - {len(subscriptions)} identified subscriptions: {json.dumps(subscriptions)}

    Please:
    1. Identify opportunities for savings
    2. Delegate research to the research agent
    3. Delegate negotiation strategies to the negotiation agent
    4. Delegate tax analysis to the tax agent
    5. Read their results and create a final report at data/final_report.md
    """

    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(block.text, end="", flush=True)

            elif isinstance(message, ResultMessage):
                logger.info(f"Duration: {message.duration_ms}ms")
                logger.info(f"Cost: ${message.total_cost_usd:.4f}")
                break

    # Step 6: Generate final report
    logger.info("Orchestration complete. Check data/final_report.md for results.")


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Financial Optimization Orchestrator Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    # Basic analysis
    uv run python financial_orchestrator.py \\
        --username john_doe \\
        --start-date 2026-01-01 \\
        --end-date 2026-01-31 \\
        --query "How can I save $500 per month?"

    # Subscription analysis
    uv run python financial_orchestrator.py \\
        --username jane_smith \\
        --start-date 2026-01-01 \\
        --end-date 2026-01-31 \\
        --query "Analyze my subscriptions and find better deals"
"""
    )

    parser.add_argument(
        "--username",
        type=str,
        required=True,
        help="Username for account (john_doe or jane_smith)"
    )

    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date in YYYY-MM-DD format"
    )

    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date in YYYY-MM-DD format"
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="User's financial question or request"
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = _parse_args()

    await _run_orchestrator(
        username=args.username,
        start_date=args.start_date,
        end_date=args.end_date,
        user_query=args.query
    )


if __name__ == "__main__":
    asyncio.run(main())
