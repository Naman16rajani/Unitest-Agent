# Repository Analyzer - Fullstack Web Application

A fullstack web application that analyzes GitHub repositories and generates unit tests using AI. The system clones repositories, processes them with external CLI tools, and creates new branches with generated test files.

## 🏗️ Architecture

- **Backend**: Django + Django REST Framework
- **Frontend**: React (Create React App)
- **Database**: SQLite (Django default)
- **API**: RESTful API with CORS support

## 📁 Project Structure

```
oscar project 2/
├── backend/                 # Django backend
│   ├── backend_project/    # Django project settings
│   ├── repo_analyzer/      # Main Django app
│   ├── manage.py           # Django management script
│   ├── requirements.txt    # Python dependencies
│   └── README.md           # Backend documentation
├── frontend/               # React frontend
│   ├── src/                # React source code
│   ├── public/             # Static assets
│   ├── package.json        # Node.js dependencies
│   └── README.md           # Frontend documentation
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- Git

### Backend Setup

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Create virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run migrations**:
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

5. **Start Django server**:
   ```bash
   python manage.py runserver
   ```

   Backend will be available at: http://localhost:8000

### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start React development server**:
   ```bash
   npm start
   ```

   Frontend will be available at: http://localhost:3000

## 🔧 Features

### Backend (Django)

- **Repository Processing**: Clone GitHub repos and process with CLI tools
- **Task Management**: Track analysis tasks with status updates
- **Query Logging**: Log all AI queries and responses
- **Git Integration**: Create branches, commit changes, push to GitHub
- **REST API**: Full REST API for frontend integration
- **Admin Interface**: Django admin for managing tasks and logs

### Frontend (React)

- **Modern UI**: Beautiful, responsive design with glassmorphism effects
- **Repository Submission**: Easy form to submit GitHub repository URLs
- **Real-time Status**: Live updates on task progress with auto-polling
- **Query History**: View all AI queries and responses for each task
- **Responsive Design**: Works seamlessly on desktop and mobile devices

## 📡 API Endpoints

- `POST /api/submit-repo/` - Submit a repository for analysis
- `GET /api/task-status/<task_id>/` - Get task status and details
- `GET /api/query-history/<task_id>/` - Get query history for a task

## 🗄️ Database Models

### RepoTask
- Tracks repository analysis tasks
- Status: pending, running, completed, failed
- Stores branch name and timestamps

### QueryLog
- Logs AI queries and responses
- Links to RepoTask via ForeignKey
- Tracks success/error status

## 🔄 Workflow

1. **Submit Repository**: User submits GitHub repo URL via frontend
2. **Task Creation**: Backend creates RepoTask with 'pending' status
3. **Repository Cloning**: System clones repo to `/tmp/repos/<task_id>/`
4. **CLI Processing**: External CLI tool analyzes code and generates tests
5. **Validation**: Generated test files are validated using py_compile
6. **Git Operations**: New branch created, tests committed and pushed
7. **Status Update**: Task status updated to 'completed'
8. **Query Logging**: All operations logged to QueryLog

## 🛠️ External CLI Tool

The backend expects an external CLI tool (`run_agent.py`) that:
- Takes a repository path as argument: `--repo /path/to/repo`
- Generates unit test files in `/tests/` directory
- Outputs results to stdout/stderr
- Backend handles validation, Git operations, and logging

## 🌐 CORS Configuration

- CORS enabled for development (allows all origins)
- Configured for frontend-backend communication
- Can be restricted for production deployment

## 📱 Responsive Design

- Mobile-first approach
- Glassmorphism UI effects
- Smooth animations and transitions
- Optimized for all device sizes

## 🚀 Deployment

### Backend
- Can be deployed to any Python hosting service
- Update CORS settings for production
- Configure database for production use

### Frontend
- Build with `npm run build`
- Deploy static files to any web server
- Update API base URL for production

## 🔍 Development

- **Hot Reload**: Frontend automatically refreshes on changes
- **Django Admin**: Access at `/admin/` for database management
- **API Testing**: Use tools like Postman or curl for API testing
- **Logging**: Comprehensive logging of all operations

## 📝 Notes

- Uses GitPython for Git operations
- Validates generated test files with py_compile
- Creates timestamped branches: `ai-unit-tests-YYYYMMDD_HHMMSS`
- Handles errors gracefully and logs all operations
- Temporary repositories stored in `/tmp/repos/`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.
