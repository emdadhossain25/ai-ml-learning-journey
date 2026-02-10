# 🔍 AI Code Reviewer

**Day 32/100 - Your personal code quality assistant**

## What It Does

AI-powered code review that catches:
- 🐛 Bugs and errors
- 🔒 Security vulnerabilities  
- 🎨 Style issues
- ⚡ Performance problems
- ✅ Best practice violations

## Features

**Multi-Language Support:**
Python, JavaScript, TypeScript, Java, C++, C, Go, Rust, Ruby, PHP, Swift, Kotlin, Bash, SQL

**Focus Modes:**
- `all` - Complete review
- `bugs` - Bugs and errors only
- `security` - Security vulnerabilities
- `style` - Code style and readability
- `performance` - Optimization opportunities

**Review Modes:**
- Single file review
- Batch review (multiple files)
- Code snippet review
- Interactive CLI

## Quick Start
```bash
# Single file
python3 code_reviewer.py mycode.py

# With focus
python3 code_reviewer.py mycode.py security

# Interactive mode
python3 code_reviewer.py
```

## Example Output
```
📝 CODE REVIEW: test_bad.py
Language: Python
Lines: 24
Focus: ALL
============================================================

1. **Critical Issues**
   - Line 5: Hardcoded password (HIGH severity)
   - Line 9: No error handling for division by zero
   - Line 22: SQL injection vulnerability

2. **Security Concerns**
   - Exposed credentials in source code
   - Unvalidated SQL query enables injection
   - No input sanitization

3. **Performance Optimizations**
   - Line 13: Use list comprehension instead of loop
   - Remove redundant boolean comparison

4. **Overall Rating**
   Score: 3/10
   Summary: Critical security and error handling issues...
```

## Use Cases

**Daily Development:**
- Review before commit
- Catch bugs early
- Learn best practices

**Code Review:**
- Pre-review assistance
- Consistent feedback
- Focus on critical issues

**Learning:**
- Understand anti-patterns
- See better alternatives
- Improve coding skills

## Cost

~$0.005 per review (Azure free tier)

## Tech Stack

- Python 3.9+
- Azure OpenAI (GPT-4o-mini)
- 300 lines of code

## Built By

Emdad Hossain | Day 32/100 Days of Code
