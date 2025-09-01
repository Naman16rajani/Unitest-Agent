# Django Backend - Repository Analyzer

This is the Django backend for the Repository Analyzer application that processes GitHub repositories and generates unit tests using an external CLI tool.

## Features

- **Repository Processing**: Clone GitHub repos and process them with CLI tools
- **Task Management**: Track repository analysis tasks with status updates
- **Query Logging**: Log all AI queries and responses
- **Git Integration**: Create branches, commit changes, and push to GitHub
- **REST API**: Full REST API for frontend integration

## Setup

1. **Create Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Migrations**:
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

4. **Create Superuser** (optional):
   ```bash
   python manage.py createsuperuser
   ```

5. **Run Development Server**:
   ```bash
   python manage.py runserver
   ```

## API Endpoints

- `POST /api/submit-repo/` - Submit a repository for analysis
- `GET /api/task-status/<task_id>/` - Get task status and details
- `GET /api/query-history/<task_id>/` - Get query history for a task

## Models

### RepoTask
- Tracks repository analysis tasks
- Status: pending, running, completed, failed
- Stores branch name and timestamps

### QueryLog
- Logs AI queries and responses
- Links to RepoTask via ForeignKey
- Tracks success/error status

## Configuration

- **CORS**: Enabled for development (allows all origins)
- **Database**: SQLite (default Django database)
- **Temp Directory**: `/tmp/repos/<task_id>/` for cloned repositories

## External CLI Tool

The backend expects an external CLI tool (`run_agent.py`) that:
- Takes a repository path as argument
- Generates unit test files in `/tests/` directory
- Outputs results to stdout/stderr

## Development Notes

- Uses GitPython for Git operations
- Validates generated test files with py_compile
- Creates timestamped branches: `ai-unit-tests-YYYYMMDD_HHMMSS`
- Handles errors gracefully and logs all operations
