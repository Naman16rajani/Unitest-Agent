from django.contrib import admin
from .models import RepoTask, QueryLog


@admin.register(RepoTask)
class RepoTaskAdmin(admin.ModelAdmin):
    list_display = ['id', 'repo_url', 'status', 'branch_name', 'created_at', 'updated_at']
    list_filter = ['status', 'created_at']
    search_fields = ['repo_url', 'branch_name']
    readonly_fields = ['id', 'created_at', 'updated_at']
    ordering = ['-created_at']


@admin.register(QueryLog)
class QueryLogAdmin(admin.ModelAdmin):
    list_display = ['id', 'task', 'status', 'timestamp']
    list_filter = ['status', 'timestamp']
    search_fields = ['input_query', 'response']
    readonly_fields = ['id', 'timestamp']
    ordering = ['-timestamp']
