import uuid
from django.db import models
from django.utils import timezone


class RepoTask(models.Model):
    """Model to track repository analysis tasks"""
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    repo_url = models.URLField(max_length=500)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    branch_name = models.CharField(max_length=255, blank=True, null=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'repo_task'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.repo_url} - {self.status}"


class QueryLog(models.Model):
    """Model to log AI queries and responses"""
    
    STATUS_CHOICES = [
        ('success', 'Success'),
        ('error', 'Error'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    task = models.ForeignKey(RepoTask, on_delete=models.CASCADE, related_name='query_logs')
    input_query = models.TextField()
    response = models.TextField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='success')
    timestamp = models.DateTimeField(default=timezone.now)
    
    class Meta:
        db_table = 'query_log'
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"{self.task.repo_url} - {self.status} - {self.timestamp}"
