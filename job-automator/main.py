"""
Job Application Automator - Main Script
Orchestrates job finding, matching, and cover letter generation
"""

import json
import os
from datetime import datetime
from cover_letter_generator import generate_cover_letter
from ai_matcher import calculate_match_score
from config import JOB_PREFERENCES, YOUR_INFO

def load_jobs():
    """Load previously saved jobs"""
    if os.path.exists('data/jobs.json'):
        with open('data/jobs.json', 'r') as f:
            return json.load(f)
    return []


def save_jobs(jobs):
    """Save jobs to file"""
    with open('data/jobs.json', 'w') as f:
        json.dump(jobs, f, indent=2)


def load_applications():
    """Load application history"""
    if os.path.exists('data/applications.json'):
        with open('data/applications.json', 'r') as f:
            return json.load(f)
    return []


def save_applications(applications):
    """Save application history"""
    with open('data/applications.json', 'w') as f:
        json.dump(applications, f, indent=2)


def add_sample_jobs():
    """
    Add sample jobs for demonstration
    In production, this would scrape from LinkedIn/BdJobs
    """
    sample_jobs = [
        {
            "id": "job_001",
            "title": "Software Engineer (Web + ML Integrations)",
            "company": "Optimizely",
            "location": "Dhaka, Bangladesh (Hybrid)",
            "posted_date": "2026-01-25",
            "source": "LinkedIn",
            "url": "https://www.linkedin.com/jobs/view/4365676286/?alternateChannel=search&eBP=NON_CHARGEABLE_CHANNEL&refId=UyrbgRWkOF6bSEzgqYYktA%3D%3D&trackingId=jXNsSmHq%2FgCjTXrAizfBpg%3D%3D&trk=d_flagship3_search_srp_jobs&lipi=urn%3Ali%3Apage%3Ad_flagship3_search_srp_jobs%3B5uPjUUh5RYiN80QphTROJw%3D%3D",
            "description": """
At Optimizely, we're on a mission to help people unlock their digital potential. We do that by reinventing how marketing and product teams work to create and optimize digital experiences across all channels. With Optimizely One, our industry-first operating system for marketers, we offer teams flexibility and choice to build their stack their way with our fully SaaS, fully decoupled, and highly composable solution.

We are proud to help more than 10,000 businesses, including H&M, PayPal, Zoom, and Toyota, enrich their customer lifetime value, increase revenue and grow their brands. Our innovation and excellence have earned us numerous recognitions as a leader by industry analysts such as Gartner, Forrester, and IDC, reinforcing our role as a trailblazer in MarTech.

At our core, we believe work is about more than just numbers -- it's about the people. Our culture is dynamic and constantly evolving, shaped by every employee, their actions and their stories. With over 1600 Optimizers spread across 12 global locations, our diverse team embodies the "One Optimizely" spirit, emphasizing collaboration and continuous improvement, while fostering a culture where every voice is heard and valued.

Join us and become part of a company that's empowering people to unlock their digital potential!

Introduction

Software Engineers in our team are hands-on builders who bridge the worlds of web engineering and intelligent automation. They understand how business goals translate into real-world integrations, crafting solutions that

connect platforms seamlessly and leverage modern AI techniques. Our engineers thrive in a fast-moving environment - designing, developing, and maintaining scalable integration systems while ensuring clean, maintainable, and secure code.

They are strong collaborators and self-starters who take ownership from concept to deployment, working closely with cross-functional teams to solve technical challenges creatively. A successful Software Engineer in this role demonstrates deep curiosity, technical excellence, and a strong ability to adapt - balancing robust engineering practices with innovative problem-solving in the areas of web systems, AI agents, and data-driven automation.

Expert Services: an organization of 150+ people within Customer Success. “Customer-first" organization.

Working Hours: Sunday to Thursday (10 AM to 6 PM), Hybrid

Responsibilities:
- Design, develop, and deploy integrations between internal and external platforms.
- Build scalable and reliable backend and frontend components using JavaScript and TypeScript.
- Work with APIs, webhooks, and data pipelines to automate workflows.
- You will design, develop, and deploy solutions using leveraging Optimizely AI platform, or more traditional builds.
- Collaborate with cross-functional teams to deliver innovative solutions and ensure seamless product integrations.
- Contribute to our DevOps and infrastructure setup, supporting deployment and monitoring best practices.


Requirements:
- 3+ years of professional software engineering experience in web or integration development.
- Strong proficiency in JavaScript and TypeScript.
- Solid understanding of Git, GitHub, and version control workflows.
- Knowledge of servers, infrastructure, and deployment pipelines.
- Practical understanding of AI and ML concepts (Agents, Agentic AI, RAG, MCP server).
- Experience in building integrations between multiple platforms (APIs, webhooks, SDKs).
- A proactive self-starter with a collaborative team mindset.
- Excellent communication skills to articulate ideas, discuss trade-offs, and collaborate with peers and stakeholders.
- Experience with cloud platforms (e.g., AWS, GCP, Azure) and containerization tools (e.g., Docker, Kubernetes)
- Experience working in agile, fast-moving environments with a focus on shipping high-quality code and robust solutions.
- Experience building, delivering, and maintaining services that comprise modern PaaS/SaaS products is a plus

Benefits:
- Best-in-class compensation plans 
- Two annual festival bonuses 
- Recognition and rewards programs 
- Vacations days 
- Annual Work/Service Anniversary Leave 
- Parental leave (both maternity and paternity) 
- Health insurance 
- Reproductive benefits for both parents 
- Volunteering opportunities to make a difference 
- Chance to work alongside our incredible global team 
- Free communal transport facilities inside Dhaka to and from the office 
- Free catered lunch every day 
"""
        }
#         ,
#         {
#             "id": "job_002",
#             "title": "ML Engineer",
#             "company": "Reve Systems",
#             "location": "Remote",
#             "posted_date": "2026-01-29",
#             "source": "LinkedIn",
#             "url": "https://www.linkedin.com/jobs/job-002",
#             "description": """
# ML Engineer position for AI-powered customer service platform.

# What you'll do:
# - Build NLP models for chatbots
# - Deploy ML models as APIs
# - Work on sentiment analysis systems
# - Optimize model performance

# Requirements:
# - Python, ML frameworks
# - NLP experience
# - API development (Flask/FastAPI)
# - Remote work capability

# Bonus:
# - Deployed ML APIs before
# - Portfolio of ML projects
# """
#         },
#         {
#             "id": "job_003",
#             "title": "Junior Data Analyst",
#             "company": "Generic Corp",
#             "location": "Dhaka (On-site only)",
#             "posted_date": "2026-01-28",
#             "source": "BdJobs",
#             "url": "https://www.bdjobs.com/job-003",
#             "description": """
# Entry-level data analyst position.

# Requirements:
# - Fresh graduate
# - Basic Excel, PowerPoint
# - Willingness to learn

# Salary: 25,000 BDT/month
# On-site: 9 AM - 6 PM, 6 days/week
# """
#         }
    ]
    
    return sample_jobs


def process_job(job):
    """
    Process a single job:
    1. Calculate match score
    2. Generate cover letter if good match
    3. Save results
    """
    print(f"\n{'=' * 60}")
    print(f"PROCESSING: {job['title']} at {job['company']}")
    print('=' * 60)
    
    # Step 1: Calculate match
    print("\n🎯 Calculating match score...")
    match_result = calculate_match_score(
        job['title'],
        job['description'],
        job['company']
    )
    
    job['match_score'] = match_result['score']
    job['match_reasons'] = match_result['reasons']
    job['red_flags'] = match_result['red_flags']
    job['recommendation'] = match_result['recommendation']
    
    print(f"   Score: {match_result['score']}/100")
    print(f"   Recommendation: {match_result['recommendation']}")
    
    # Step 2: Generate cover letter if score >= threshold
    if match_result['score'] >= JOB_PREFERENCES['min_match_score']:
        print("\n✍️  Generating cover letter...")
        
        cover_letter = generate_cover_letter(
            job['title'],
            job['company'],
            job['description'],
            "\n".join(f"- {r}" for r in match_result['reasons'])
        )
        
        job['cover_letter'] = cover_letter
        job['status'] = "Ready to apply"
        
        # Save cover letter to file
        filename = f"data/cover_letter_{job['company'].replace(' ', '_')}_{job['id']}.txt"
        with open(filename, 'w') as f:
            f.write(cover_letter)
        
        print(f"   ✅ Cover letter saved: {filename}")
    else:
        print(f"\n   ⏭️  Skipping (score below threshold of {JOB_PREFERENCES['min_match_score']})")
        job['cover_letter'] = None
        job['status'] = "Skipped - low match"
    
    job['processed_date'] = datetime.now().isoformat()
    
    return job


def generate_daily_digest(processed_jobs):
    """Generate summary of today's job findings"""
    
    high_match = [j for j in processed_jobs if j['match_score'] >= 80]
    medium_match = [j for j in processed_jobs if 60 <= j['match_score'] < 80]
    low_match = [j for j in processed_jobs if j['match_score'] < 60]
    
    digest = f"""
{'=' * 60}
DAILY JOB SEARCH DIGEST - {datetime.now().strftime('%Y-%m-%d')}
{'=' * 60}

📊 SUMMARY:
   Total jobs analyzed: {len(processed_jobs)}
   High match (80-100): {len(high_match)}
   Medium match (60-79): {len(medium_match)}
   Low match (0-59): {len(low_match)}

{'=' * 60}
🎯 HIGH PRIORITY (80-100% match)
{'=' * 60}
"""
    
    for job in high_match:
        digest += f"""
{job['title']} at {job['company']}
   Score: {job['match_score']}/100
   Location: {job['location']}
   URL: {job['url']}
   Status: {job['status']}
   
   Top Reasons:
"""
        for reason in job['match_reasons'][:3]:
            digest += f"   ✓ {reason}\n"
        
        if job['cover_letter']:
            digest += f"   📄 Cover letter: data/cover_letter_{job['company'].replace(' ', '_')}_{job['id']}.txt\n"
        digest += "\n"
    
    if medium_match:
        digest += f"\n{'=' * 60}\n⚠️  MEDIUM PRIORITY (60-79% match)\n{'=' * 60}\n"
        for job in medium_match:
            digest += f"\n{job['title']} at {job['company']} - {job['match_score']}/100\n"
            digest += f"   {job['url']}\n"
    
    digest += f"\n{'=' * 60}\n"
    digest += f"Next Steps:\n"
    digest += f"1. Review {len(high_match)} high-priority cover letters\n"
    digest += f"2. Customize and submit applications\n"
    digest += f"3. Check {len(medium_match)} medium-priority jobs\n"
    digest += f"{'=' * 60}\n"
    
    return digest


def main():
    """Main automation workflow"""
    print("=" * 60)
    print("JOB APPLICATION AUTOMATOR - DAY 23")
    print("=" * 60)
    
    print("\n🔍 Step 1: Loading jobs...")
    # In production, this would scrape LinkedIn/BdJobs
    # For now, using sample jobs
    jobs = add_sample_jobs()
    print(f"   Found {len(jobs)} new jobs")
    
    print("\n🤖 Step 2: Processing jobs with AI...")
    processed_jobs = []
    
    for job in jobs:
        try:
            processed_job = process_job(job)
            processed_jobs.append(processed_job)
        except Exception as e:
            print(f"   ❌ Error processing {job['title']}: {e}")
            continue
    
    print("\n💾 Step 3: Saving results...")
    save_jobs(processed_jobs)
    print("   ✅ Jobs saved to: data/jobs.json")
    
    print("\n📧 Step 4: Generating daily digest...")
    digest = generate_daily_digest(processed_jobs)
    
    # Save digest
    digest_filename = f"data/digest_{datetime.now().strftime('%Y%m%d')}.txt"
    with open(digest_filename, 'w') as f:
        f.write(digest)
    
    print(f"   ✅ Digest saved to: {digest_filename}")
    
    print("\n" + digest)
    
    print("\n" + "=" * 60)
    print("✅ AUTOMATION COMPLETE!")
    print("=" * 60)
    print(f"""
WHAT JUST HAPPENED:
1. Analyzed {len(jobs)} job postings
2. AI calculated match scores for each
3. Generated cover letters for high-match jobs
4. Saved everything to data/ folder
5. Created daily digest

NEXT STEPS:
1. Review cover letters in data/ folder
2. Customize if needed (AI gets you 80% there!)
3. Apply to high-match jobs
4. Track applications in your tracker

Time saved: ~{len([j for j in processed_jobs if j.get('cover_letter')])  * 20} minutes (vs manual)
""")


if __name__ == "__main__":
    main()
