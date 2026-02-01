"""
AI Job Matcher - Scores jobs based on skill match
"""

from openai import OpenAI

from config import OPENAI_API_KEY, YOUR_SKILLS, YOUR_EXPERIENCE

client = OpenAI(api_key=OPENAI_API_KEY)


def calculate_match_score(job_title, job_description, company_name):
    """
    Use AI to calculate how well a job matches your profile
    
    Returns:
        dict with score (0-100) and reasons
    """

    prompt = f"""
Analyze this job posting and determine how well it matches this candidate's profile.

JOB POSTING:
Title: {job_title}
Company: {company_name}
Description: {job_description[:800]}

CANDIDATE PROFILE:
Skills: {', '.join(YOUR_SKILLS)}
Experience: {YOUR_EXPERIENCE}

TASK:
1. Calculate a match score from 0-100 where:
   - 90-100: Perfect match, must apply
   - 70-89: Strong match, should apply
   - 50-69: Decent match, consider applying
   - Below 50: Not a good fit

2. List 3-5 specific reasons WHY this is/isn't a good match

3. Identify any red flags (unrealistic requirements, low pay indicators, etc)

Respond in this EXACT format:
SCORE: [number 0-100]
REASONS:
- [Reason 1]
- [Reason 2]
- [Reason 3]
RED_FLAGS:
- [Flag 1 or "None"]
RECOMMENDATION: [Apply/Consider/Skip]
"""

    try:
        response = client.chat.completions.create(model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert recruiter who evaluates job-candidate fit objectively and helps candidates focus on the best opportunities."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=400,
        temperature=0.3)

        result = response.choices[0].message.content.strip()

        # Parse the result
        score = extract_score(result)
        reasons = extract_reasons(result)
        red_flags = extract_red_flags(result)
        recommendation = extract_recommendation(result)

        return {
            "score": score,
            "reasons": reasons,
            "red_flags": red_flags,
            "recommendation": recommendation,
            "raw_analysis": result
        }

    except Exception as e:
        print(f"Error in AI matching: {e}")
        return fallback_match(job_title, job_description)


def extract_score(text):
    """Extract numerical score from AI response"""
    import re
    match = re.search(r'SCORE:\s*(\d+)', text)
    if match:
        return int(match.group(1))
    return 50  # Default if parsing fails


def extract_reasons(text):
    """Extract reasons list from AI response"""
    import re
    reasons_section = re.search(r'REASONS:(.*?)(?:RED_FLAGS:|$)', text, re.DOTALL)
    if reasons_section:
        reasons_text = reasons_section.group(1)
        reasons = re.findall(r'-\s*(.+)', reasons_text)
        return reasons[:5]  # Max 5 reasons
    return ["Unable to parse reasons"]


def extract_red_flags(text):
    """Extract red flags from AI response"""
    import re
    flags_section = re.search(r'RED_FLAGS:(.*?)(?:RECOMMENDATION:|$)', text, re.DOTALL)
    if flags_section:
        flags_text = flags_section.group(1)
        flags = re.findall(r'-\s*(.+)', flags_text)
        return flags if flags else ["None"]
    return ["None"]


def extract_recommendation(text):
    """Extract recommendation from AI response"""
    import re
    match = re.search(r'RECOMMENDATION:\s*(\w+)', text)
    if match:
        return match.group(1)
    return "Consider"


def fallback_match(job_title, job_description):
    """Simple keyword-based fallback if AI fails"""
    score = 0
    reasons = []

    job_text = (job_title + " " + job_description).lower()

    # Check for key skills
    ml_keywords = ['machine learning', 'ml', 'ai', 'deep learning', 'tensorflow', 'scikit']
    python_keywords = ['python', 'pandas', 'numpy']
    senior_keywords = ['senior', 'lead', 'experienced', '5+ years', '10+ years']

    if any(kw in job_text for kw in ml_keywords):
        score += 40
        reasons.append("ML/AI role - matches your specialization")

    if any(kw in job_text for kw in python_keywords):
        score += 20
        reasons.append("Python required - you have this skill")

    if any(kw in job_text for kw in senior_keywords):
        score += 20
        reasons.append("Senior role - matches your 15 years experience")

    if 'remote' in job_text or 'work from home' in job_text:
        score += 10
        reasons.append("Remote-friendly - your preference")

    if 'bangladesh' in job_text or 'dhaka' in job_text:
        score += 10
        reasons.append("Bangladesh-based - local opportunity")

    raw_text = f"""
SCORE: {min(score, 100)}
REASONS:
{chr(10).join('- ' + r for r in reasons) if reasons else '- Match based on keywords'}
RED_FLAGS:
- None
RECOMMENDATION: {'Apply' if score >= 70 else 'Consider' if score >= 50 else 'Skip'}
"""

    return {
        "score": min(score, 100),
        "reasons": reasons if reasons else ["Match based on keywords"],
        "red_flags": ["None"],
        "recommendation": "Apply" if score >= 70 else "Consider" if score >= 50 else "Skip",
        "raw_analysis": raw_text  # ADD THIS LINE!
    }

def test_matcher():
    """Test the job matcher"""
    print("=" * 60)
    print("TESTING AI JOB MATCHER")
    print("=" * 60)

    test_jobs = [
        {
            "title": "Senior Machine Learning Engineer",
            "company": "TechCorp Bangladesh",
            "description": """
We're looking for a Senior ML Engineer to join our AI team.

Requirements:
- 5+ years software engineering experience
- Python, TensorFlow, scikit-learn
- Production ML deployment experience
- Team collaboration skills
- Remote work OK

Responsibilities:
- Build and deploy ML models
- Lead ML projects
- Mentor junior engineers
- Work with product team
"""
        },
        {
            "title": "Junior Data Analyst",
            "company": "StartupXYZ",
            "description": """
Entry-level data analyst position.

Requirements:
- Fresh graduate
- Basic Excel skills
- Willingness to learn

Note: On-site only, Dhaka office
"""
        }
    ]

    for i, job in enumerate(test_jobs, 1):
        print(f"\n{'=' * 60}")
        print(f"JOB {i}: {job['title']} at {job['company']}")
        print('=' * 60)
        print("\nAnalyzing... (takes 5-10 seconds)")

        match_result = calculate_match_score(
            job['title'],
            job['description'],
            job['company']
        )

        print(f"\n🎯 MATCH SCORE: {match_result['score']}/100")
        print(f"📋 RECOMMENDATION: {match_result['recommendation']}")

        print(f"\n✅ REASONS:")
        for reason in match_result['reasons']:
            print(f"   • {reason}")

        print(f"\n⚠️  RED FLAGS:")
        for flag in match_result['red_flags']:
            print(f"   • {flag}")

        print(f"\nRaw AI Analysis:")
        if 'raw_analysis' in match_result:
            print(match_result['raw_analysis'])
        else:
            print("(Used fallback matching - no detailed analysis)")

    print("\n" + "=" * 60)
    print("✅ JOB MATCHER TEST COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    test_matcher()
