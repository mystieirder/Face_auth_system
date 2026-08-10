fetch('https://openrouter.ai/api/v1/chat/completions', {
  method: 'POST',
  headers: {
    Authorization: 'Bearer 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',
    'HTTP-Referer': '<YOUR_SITE_URL>',
    'X-Title': '<YOUR_SITE_NAME>',
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'openai/gpt-4o',
    messages: [
      {
        role: 'user',
        content: 'What is the meaning of life?',
      },
    ],
  }),
});