from django.shortcuts import render
from rest_framework import viewsets
from .models import Agent
from .serializers import AgentSerializer
from .agent import AgentExecuter
from asgiref.sync import sync_to_async
from rest_framework.response import Response
import asyncio





# Create your views here.
class AgentViewset(viewsets.ViewSet):
    serailizer_class = AgentSerializer()
    queryset= Agent.objects.all()


    def agent_query(self, request):
        query = request.data.get("query", None)

        if query is None:
            return Response({
                "error":"query should be available"
            }, status=400)

        agent=AgentExecuter()

        try:
            x,result = agent.query(query)
            print(result)
            return Response({
                "result":result
            }, x)
        except Exception as e:

            return Response({
                "error":e
            }, 400)

