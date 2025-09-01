from django.urls import path
from .views import SubmitRepoView, TaskStatusView, QueryHistoryView

app_name = 'repo_analyzer'

urlpatterns = [
    path('submit-repo/', SubmitRepoView.as_view(), name='submit-repo'),
    path('task-status/<uuid:task_id>/', TaskStatusView.as_view(), name='task-status'),
    path('query-history/<uuid:task_id>/', QueryHistoryView.as_view(), name='query-history'),
]
