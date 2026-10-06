from pprint import pprint

from agent.parser import parse_request
from agent.planner import plan_request


TEST_CASES = [
    {
        "name": "Price prediction",
        "request": "What is the tomato price in Maharashtra?",
    },
    {
        "name": "Price forecast",
        "request": "What will tomato prices be in Maharashtra?",
    },
    {
        "name": "Weather",
        "request": "What is the weather in Maharashtra?",
    },
    {
        "name": "Disease detection",
        "request": "What disease is affecting my tomato leaf?",
    },
    {
        "name": "Price + weather",
        "request": "What is the tomato price and weather in Maharashtra?",
    },
    {
        "name": "Disease + advice",
        "request": "My tomato has disease. What should I do?",
    },
        {
        "name": "Agriculture advice",
        "request": "What should I do about my tomato disease?",
    },
    {
        "name": "Disease treatment",
        "request": "My tomato leaf is infected. What treatment should I use?",
    },
]


def main():

    print("=" * 80)
    print("CROPBALANCE AGENT — PARSER + PLANNER AUDIT")
    print("=" * 80)

    for i, case in enumerate(TEST_CASES, start=1):

        print(f"\n{'=' * 80}")
        print(f"TEST {i}: {case['name']}")
        print(f"{'=' * 80}")

        print(f"Request:")
        print(f"  {case['request']}")

        parsed = parse_request(case["request"])
        plan = plan_request(parsed)

        print("\nParsed:")
        pprint(parsed)

        print("\nPlan:")
        pprint(plan)

        print("\nTools selected:")
        print(f"  {plan.get('tools', [])}")


if __name__ == "__main__":
    main()