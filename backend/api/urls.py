from django.urls import path
from .views import AgentViewset


urlpatterns=[
    path('ask', AgentViewset.as_view({'post':'agent_query'}))
]