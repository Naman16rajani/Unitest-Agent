from rest_framework import serializers
from .models import RepoTask, QueryLog


class RepoTaskSerializer(serializers.ModelSerializer):
    """Serializer for RepoTask model"""
    
    class Meta:
        model = RepoTask
        fields = ['id', 'repo_url', 'status', 'branch_name', 'created_at', 'updated_at']
        read_only_fields = ['id', 'status', 'branch_name', 'created_at', 'updated_at']


class QueryLogSerializer(serializers.ModelSerializer):
    """Serializer for QueryLog model"""
    
    class Meta:
        model = QueryLog
        fields = ['id', 'task', 'input_query', 'response', 'status', 'timestamp']
        read_only_fields = ['id', 'timestamp']


class RepoTaskDetailSerializer(serializers.ModelSerializer):
    """Detailed serializer for RepoTask with query count"""
    total_queries = serializers.SerializerMethodField()
    
    class Meta:
        model = RepoTask
        fields = ['id', 'repo_url', 'status', 'branch_name', 'created_at', 'updated_at', 'total_queries']
    
    def get_total_queries(self, obj):
        return obj.query_logs.count()


class RepoSubmitSerializer(serializers.Serializer):
    """Serializer for repository submission"""
    repo_url = serializers.URLField(max_length=500)
