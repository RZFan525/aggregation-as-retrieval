| Method | Model | Input/problem (k) | Output/problem (k) | Total/problem (k) | Notes |
|---|---|---|---|---|---|
| Candidate generation (8 samples, reference) | 2B | 0.0 | 150.4 | 150.4 |  |
| MV@8 | 2B | 0.0 | 0.0 | 0.0 |  |
| RAG-BM25 (m=4) | 2B | 4.3 | 12.2 | 16.5 |  |
| RAG-BM25 (m=8) | 2B | 8.3 | 10.8 | 19.1 |  |
| RAG-BM25 (m=16) | 2B | 16.3 | 8.7 | 25.0 |  |
| RAG-BM25 (m=32) | 2B | 32.2 | 8.8 | 41.0 |  |
| RAG-BM25 (m=64) | 2B | 62.8 | 8.8 | 71.6 |  |
| RAG-BM25 (m=128) | 2B | 100.1 | 9.5 | 109.7 |  |
| RAG-Qwen3Emb (m=16) | 2B | 16.4 | 8.1 | 24.5 |  |
| ICA Full-CoT | 2B | 150.7 | 12.4 | 163.2 |  |
| Agentic | 2B | 413.7 | 21.8 | 435.5 | main loop 346.4k in / 18.7k out; 3.7 summary calls/problem ≈ 67.4k in / ≥3.0k out (estimated) |
| Candidate generation (8 samples, reference) | 4B | 0.0 | 228.1 | 228.1 |  |
| MV@8 | 4B | 0.0 | 0.0 | 0.0 |  |
| RAG-BM25 (m=4) | 4B | 4.3 | 21.0 | 25.2 |  |
| RAG-BM25 (m=8) | 4B | 8.3 | 19.3 | 27.6 |  |
| RAG-BM25 (m=16) | 4B | 16.4 | 17.3 | 33.7 |  |
| RAG-BM25 (m=32) | 4B | 32.6 | 15.2 | 47.8 |  |
| RAG-BM25 (m=64) | 4B | 65.0 | 13.7 | 78.6 |  |
| RAG-BM25 (m=128) | 4B | 127.6 | 12.6 | 140.2 |  |
| RAG-Qwen3Emb (m=16) | 4B | 16.5 | 17.8 | 34.3 |  |
| ICA Full-CoT | 4B | 213.4 | 10.9 | 224.3 |  |
| Agentic | 4B | 315.7 | 14.0 | 329.6 | main loop 151.9k in / 11.9k out; 5.3 summary calls/problem ≈ 163.8k in / ≥2.1k out (estimated) |
| Candidate generation (8 samples, reference) | 9B | 0.0 | 220.8 | 220.8 |  |
| MV@8 | 9B | 0.0 | 0.0 | 0.0 |  |
| RAG-BM25 (m=4) | 9B | 4.3 | 19.0 | 23.2 |  |
| RAG-BM25 (m=8) | 9B | 8.3 | 17.1 | 25.4 |  |
| RAG-BM25 (m=16) | 9B | 16.4 | 14.7 | 31.2 |  |
| RAG-BM25 (m=32) | 9B | 32.6 | 12.8 | 45.4 |  |
| RAG-BM25 (m=64) | 9B | 64.9 | 11.0 | 75.9 |  |
| RAG-BM25 (m=128) | 9B | 126.8 | 9.0 | 135.9 |  |
| RAG-Qwen3Emb (m=16) | 9B | 16.5 | 15.0 | 31.6 |  |
| ICA Full-CoT | 9B | 208.5 | 9.3 | 217.8 |  |
| Agentic | 9B | 331.5 | 11.8 | 343.2 | main loop 159.9k in / 8.2k out; 6.0 summary calls/problem ≈ 171.5k in / ≥3.5k out (estimated) |
