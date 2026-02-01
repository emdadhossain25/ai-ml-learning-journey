# 🤖 AI-Powered Job Application Automator

**Automate your job search with AI!**

Built on Day 23 of 100DaysOfCode journey by Emdad Hossain.

## What It Does

1. **Finds Jobs** - Scrapes LinkedIn, BdJobs for ML/AI positions
2. **AI Matching** - Scores each job (0-100) based on your skills
3. **Cover Letters** - AI generates customized letters for good matches
4. **Tracking** - Saves everything for review
5. **Daily Digest** - Summary email of opportunities

## Tech Stack

- **AI:** OpenAI GPT-3.5-turbo
- **Language:** Python
- **Automation:** GitHub Actions (runs daily)
- **Storage:** JSON files (can upgrade to database)

## Setup

1. Clone repo
2. Install dependencies:
```bash
pip install openai --break-system-packages
```

3. Add your OpenAI API key to `config.py`

4. Run:
```bash
python3 main.py
```

## Features

✅ **AI Match Scoring** - Smart analysis of job fit  
✅ **Auto Cover Letters** - 80% done, you customize 20%  
✅ **Red Flag Detection** - Warns about suspicious postings  
✅ **Daily Digest** - Summary of best opportunities  
✅ **Time Savings** - 20 min → 2 min per application  

## Example Output
```
Job: Senior ML Engineer at Brain Station 23
Match Score: 92/100
Reasons:
✓ 15 years experience matches "senior" requirement
✓ Production ML deployment experience
✓ Team leadership capability
✓ Bangladesh-based company

Cover letter generated ✅
Status: Ready to apply
```

## Business Value

**Problem:** Job searching is time-consuming  
- Manual: 30 min per application × 20 jobs = 10 hours/week  

**Solution:** AI automation  
- Automated: 2 min review × 20 jobs = 40 minutes/week  
- **Time saved: 9+ hours/week!** 🚀

## Future Enhancements

- [ ] Real scraping (LinkedIn, BdJobs APIs)
- [ ] Email notifications
- [ ] Application submission automation
- [ ] Interview prep generator
- [ ] Salary negotiation assistant
- [ ] GitHub Actions cron (daily runs)

## Author

**Emdad Hossain**  
Senior Software Engineer → ML Engineer  
- Portfolio: https://emdadhossain25.github.io/emdad-portfolio/
- Kaggle: https://www.kaggle.com/emdadhossain25
- GitHub: https://github.com/emdadhossain25

## License

MIT - Feel free to use and modify!
