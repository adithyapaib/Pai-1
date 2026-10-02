"""Send live curl-equivalent requests to the local Pai-1 API."""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_URL = "http://127.0.0.1:8000/v1/decide"
EVALUATE_URL = "http://127.0.0.1:8000/v1/evaluate"


def post_json(url: str, payload: dict) -> dict:
    """POST JSON and return the decoded response."""
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            print(f"HTTP {response.status}")
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        print(f"HTTP {error.code}: {detail}")
        raise
    except URLError as error:
        print(f"Could not connect to {url}: {error.reason}")
        raise

test_cases = [
    {
        "state": "The production API is returning HTTP 500 errors after the latest deployment.",
        "question": "What should the engineering team do first?",
        "options": {
            "investigate": "Investigate logs and identify the root cause.",
            "rollback": "Rollback the latest deployment.",
            "escalate": "Escalate the incident to the senior engineering team.",
        },
    },
    {
        "state": "A database migration is scheduled for tonight and affects customer tables.",
        "question": "What is the safest preparation step?",
        "options": {
            "backup": "Create and verify a database backup.",
            "skip": "Skip the migration to avoid all risk.",
            "notify": "Notify customers without preparing a recovery plan.",
        },
    },
    {
        "state": "A customer reports that their account may have been accessed by someone else.",
        "question": "What should support do next?",
        "options": {
            "secure": "Secure the account and begin an access review.",
            "ignore": "Ignore the report until more customers complain.",
            "delete": "Delete the account immediately.",
        },
    },
    {
        "state": "A team has budget for one improvement this quarter. Customer feedback shows slow page loads and confusing navigation.",
        "question": "Which improvement should be prioritized?",
        "options": {
            "performance": "Improve page-load performance and measure the result.",
            "navigation": "Redesign navigation based on usability research.",
            "marketing": "Spend the budget on a new advertising campaign.",
        },
    },
    {
        "state": "A small team needs to choose a location for a one-day planning workshop.",
        "question": "Which option is most practical?",
        "options": {
            "office": "Use a quiet meeting room at the office.",
            "park": "Hold the workshop outdoors regardless of weather.",
            "flight": "Fly the team to another country for the workshop.",
        },
    },
    {
        "state": "A new employee starts next Monday and has never used the company's internal tools.",
        "question": "What should the manager prepare first?",
        "options": {
            "onboarding": "Prepare access, documentation, and an onboarding schedule.",
            "surprise": "Give no preparation and let the employee discover everything alone.",
            "meeting": "Schedule meetings without providing tool access.",
        },
    },
]

for index, payload in enumerate(test_cases, start=1):
    print(f"\nTest case {index}: {payload['question']}")
    print(json.dumps(post_json(API_URL, payload), indent=2))


multi_question_payload = {
    "model": "clef",
    "state": "Checkout has been failing for every customer for the last hour.",
    "questions": {
        "urgent": {
            "type": "noul",
            "instructions": "Is this support request urgent?",
        },
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this request?",
            "criteria": {
                "billing": "Payments, invoices, and refunds",
                "technical": "Outages, errors, and configuration",
                "sales": "Plans and upgrades",
            },
        },
        "severity": {
            "type": "score",
            "instructions": "How severe is the customer impact?",
            "criteria": ["No impact", "Minor", "Major", "Critical"],
        },
    },
}

print("\nMulti-question evaluation")
print(json.dumps(post_json(EVALUATE_URL, multi_question_payload), indent=2))