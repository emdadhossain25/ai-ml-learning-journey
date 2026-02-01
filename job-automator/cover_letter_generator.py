"""
AI-Powered Cover Letter Generator
Uses OpenAI GPT to generate customized cover letters
"""

from openai import OpenAI

from config import OPENAI_API_KEY, YOUR_INFO, YOUR_EXPERIENCE

client = OpenAI(api_key = OPENAI_API_KEY)



def generate_cover_letter(job_title, company_name, job_description, match_reasons):
    """
    Generate customized cover letter using AI
    
    Args:
        job_title: Title of the job
        company_name: Name of the company
        job_description: Full job description
        match_reasons: Why you're a good match (from AI matcher)
    
    Returns:
        Generated cover letter text
    """

    prompt = f"""
Write a professional, compelling cover letter for this job application:

JOB TITLE: {job_title}
COMPANY: {company_name}

JOB DESCRIPTION:
{job_description[:1000]}  # Limit to avoid token limits

CANDIDATE INFORMATION:
Name: {YOUR_INFO['name']}
Background: {YOUR_EXPERIENCE}
Portfolio: {YOUR_INFO['portfolio']}
GitHub: {YOUR_INFO['github']}
Kaggle: {YOUR_INFO['kaggle']}

WHY I'M A GOOD MATCH:
{match_reasons}

INSTRUCTIONS:
1. Keep it concise (200-250 words)
2. Start with why I'm excited about THIS SPECIFIC company
3. Highlight my unique combination: 15 years software engineering + fresh ML skills
4. Mention 2-3 relevant projects (churn prediction, sentiment API, Kaggle)
5. Emphasize production deployment capability (not just notebooks)
6. Include specific numbers (99.9% crash-free, 10K users, 22 days learning)
7. End with clear call to action
8. Professional but authentic tone
9. Bangladesh context (remote-friendly, local companies)
10. Show enthusiasm but not desperation

Generate the cover letter now:
"""

    try:
        response = client.chat.completions.create(model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert career coach who writes compelling, personalized cover letters that get interviews. You emphasize unique value propositions and concrete achievements."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7)

        cover_letter = response.choices[0].message.content.strip()
        return cover_letter

    except Exception as e:
        print(f"Error generating cover letter: {e}")
        return generate_fallback_cover_letter(job_title, company_name)


def generate_fallback_cover_letter(job_title, company_name):
    """Fallback template if AI fails"""
    return f"""Dear Hiring Manager at {company_name},

I am writing to express my strong interest in the {job_title} position at {company_name}.

With 15 years of software engineering experience and recent intensive specialization in Machine Learning, I bring a unique combination of production expertise and cutting-edge ML skills. At Mir Info Systems, I led a team of 5 engineers and delivered production applications serving 10,000+ users with 99.9% crash-free sessions.

Over the past 22 days, I've built 18+ ML projects including:
- Production sentiment analysis API (deployed and live)
- Customer churn prediction system (85% accuracy, $300K business impact)
- Deep learning image classifier (99.3% accuracy using transfer learning)
- Active participation in 2 Kaggle competitions

What sets me apart is my ability to ship production ML systems, not just experiment in notebooks. I've deployed REST APIs, built full-stack applications, and led teams - skills that bridge the gap between ML research and real-world deployment.

Portfolio: {YOUR_INFO['portfolio']}
GitHub: {YOUR_INFO['github']}
Kaggle: {YOUR_INFO['kaggle']}

I would welcome the opportunity to discuss how my unique combination of experience and ML expertise could contribute to {company_name}'s success.

Best regards,
{YOUR_INFO['name']}
{YOUR_INFO['email']}
{YOUR_INFO['phone']}
"""


def test_generator():
    """Test the cover letter generator"""
    print("=" * 60)
    print("TESTING AI COVER LETTER GENERATOR")
    print("=" * 60)

    # Test job
    test_job = {
        "title": "Machine Learning Engineer",
        "company": "TechCorp Bangladesh",
        "description": """
We are looking for a Machine Learning Engineer to join our team.
Responsibilities:
- Build and deploy ML models in production
- Work with large datasets
- Collaborate with product team
- Optimize model performance

Requirements:
- Python, TensorFlow, scikit-learn
- Production deployment experience
- Strong communication skills
- Remote work capability
""",
        "match_reasons": """
- 15 years software engineering experience
- Recent ML specialization (22 days intensive learning)
- Production deployment expertise (deployed sentiment API)
- Built 18+ ML projects
- Active on Kaggle
- Team leadership experience
"""
    }

    print("\nGenerating cover letter for:")
    print(f"  Job: {test_job['title']}")
    print(f"  Company: {test_job['company']}")
    print("\nGenerating... (takes 5-10 seconds)")

    cover_letter = generate_cover_letter(
        test_job['title'],
        test_job['company'],
        test_job['description'],
        test_job['match_reasons']
    )

    print("\n" + "=" * 60)
    print("GENERATED COVER LETTER:")
    print("=" * 60)
    print(cover_letter)
    print("\n" + "=" * 60)

    # Save to file
    with open('data/sample_cover_letter.txt', 'w') as f:
        f.write(cover_letter)

    print("\n✅ Cover letter saved to: data/sample_cover_letter.txt")
    print(f"✅ Length: {len(cover_letter.split())} words")


if __name__ == "__main__":
    test_generator()
