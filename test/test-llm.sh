#!/bin/bash

time curl http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm4:latest",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "你好，我是一名研一的学生，研究方向是语音合成"
      }
    ],
    "temperature": 0.5,
    "stream": false
  }'
