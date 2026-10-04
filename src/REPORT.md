# Báo cáo Day 17: Memory Systems for AI Agent

## 1. Cách chạy

```bash
.venv/Scripts/activate              # Windows (Linux/macOS: source .venv/bin/activate)
python src/benchmark.py             # 2 bảng: Standard + Long-Context Stress (offline, lặp lại được)
python src/benchmark.py --verbose   # in thêm từng câu hỏi recall và câu trả lời
pytest src/test_agents.py -v        # 21 test
```

Chế độ live (tùy chọn): tạo `.env` theo mẫu `.env.example`, đặt `LAB_MODE=live`, rồi chạy `python src/benchmark.py --live`. Hỗ trợ các provider `openai`, `custom`, `gemini`, `anthropic`, `ollama` và `openrouter`.

## 2. Kiến trúc

| Lớp | Nơi lưu | Vòng đời | Thành phần |
|---|---|---|---|
| Short-term | `CompactMemoryManager.state[thread_id]["messages"]` | trong 1 thread | giữ nguyên văn `compact_keep_messages` message gần nhất |
| Compact | `CompactMemoryManager.state[thread_id]["summary"]` | trong 1 thread | tóm tắt cuốn chiếu các message cũ, giới hạn `summary_max_items` bullet |
| Persistent | `state/profiles/<user>/User.md` | xuyên thread, xuyên process | fact có cấu trúc, confidence, log đính chính |

Một lượt của Advanced Agent chạy như sau:

`extract_profile_candidates()` → lọc theo confidence → `upsert_fact()` (xử lý xung đột) → `compact.append()` (tự compact khi vượt ngưỡng) → prompt = system + `User.md` + summary + recent → trả lời.

**Baseline Agent** giữ toàn bộ message của thread và gửi lại hết trong mỗi prompt. Nó không có `User.md` và không compact. Trong cùng một thread, nó vẫn trả lời được các fact đã nghe (test `test_cross_session_recall` có kiểm tra). Sang thread mới thì nó quên hết.

**Chế độ live:** dùng `create_agent` + `InMemorySaver` cho cả hai agent. Advanced Agent có thêm:
- `@dynamic_prompt` để inject `User.md` vào system prompt (không cần tool đọc);
- `SummarizationMiddleware` với `trigger=("tokens", 1000)`, `keep=("tokens", 300)` và một prompt tóm tắt ngắn;
- extractor có guardrail (giống bản offline) ghi `User.md` **trước** mỗi lần gọi model;
- tool `save_user_fact` cho LLM, **tắt mặc định** (`LIVE_MEMORY_TOOL=1` để bật). Lý do ở mục 3b.

Token ở chế độ live là **token thật do provider báo**, thu qua callback `UsageTracker`. Con số bao gồm mọi lần gọi LLM trong một lượt: vòng gọi tool và cả lần gọi tóm tắt "ẩn" của middleware. Cột Compactions ở live đếm số lần middleware thực sự tóm tắt.


## 3. Kết quả benchmark (offline, cấu hình mặc định)

Cấu hình: `compact_threshold_tokens=1000`, `compact_keep_messages=4`, `profile_confidence_threshold=0.6`. Token được ước lượng bằng heuristic ~4 ký tự/token.

### Standard Benchmark (`data/conversations.json`: 10 hội thoại, 101 lượt, 14 câu hỏi recall)

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|------------------:|------------------------:|---------------------:|-----------------:|----------------------:|------------:|
| Baseline |             2,627 |                  15,213 |                   0% |             0.20 |                     0 |           0 |
| Advanced |             2,692 |                  35,092 |                 100% |             1.00 |                   769 |           0 |

So với Baseline, Advanced tốn thêm **+2.5% agent tokens** và **+130.7% prompt tokens**, đổi lại recall **tăng 100 điểm %**.

### Long-Context Stress Benchmark (`data/advanced_long_context.json`: 16 lượt rất dài, 3 câu hỏi recall)

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|------------------:|------------------------:|---------------------:|-----------------:|----------------------:|------------:|
| Baseline |             2,561 |                  22,126 |                   0% |             0.20 |                     0 |           0 |
| Advanced |             2,755 |                  12,804 |                 100% |             1.00 |                   544 |           3 |

So với Baseline, Advanced tốn thêm **+7.6% agent tokens** nhưng **giảm 42.1% prompt tokens**, và recall **tăng 100 điểm %**.

### Prompt tokens theo từng lượt (stress test)

| Lượt | 1 | 3 | 5 | **6** | 9 | **10** | 13 | **14** | 16 |
|---|---|---|---|---|---|---|---|---|---|
| Baseline | 216 | 532 | 863 | 1,025 | 1,463 | 1,592 | 2,054 | 2,185 | 2,472 |
| Advanced | 326 | 664 | 1,004 | **547** | 1,053 | **615** | 1,065 | **622** | 901 |

Baseline tăng tuyến tính, nên tổng chi phí của cả hội thoại tăng theo bậc hai. Advanced có dạng răng cưa: mỗi lần chạm ngưỡng 1.000 token (ở lượt 6, 10 và 14) thì bị compact kéo về khoảng 550–620 token. Ở lượt 1–5, Advanced đắt hơn vì phải kéo theo `User.md`. Điểm giao cắt nằm ở lần compact đầu tiên.

### Độ nhạy theo ngưỡng compact (stress test)

| `COMPACT_THRESHOLD_TOKENS` | 600 | 800 | 1000 | 1200 | 2000 | 4000 (không compact) |
|---|---|---|---|---|---|---|
| Compactions | 8 | 4 | 3 | 2 | 1 | 0 |
| Prompt tokens so với Baseline | −52.3% | −46.2% | −42.1% | −34.1% | −14.6% | **+14.2%** |
| Recall | 100% | 100% | 100% | 100% | 100% | 100% |

Khi không có compact, Advanced **đắt hơn** Baseline ngay cả với hội thoại dài, vì chi phí `User.md` cộng thêm vào mỗi lượt. Vậy phần tiết kiệm đến từ compact, còn `User.md` là thứ đem lại recall.

## 3b. Kết quả chế độ live (OpenAI `gpt-4o-mini`, token thật)

Lệnh chạy: `python src/benchmark.py --live`, với `.env` có `LAB_MODE=live`. `Response quality` ở chế độ này do LLM judge chấm (cùng model `gpt-4o-mini`), quy về thang 0–1. Cấu hình mặc định: ngưỡng compact 1000 và tắt tool.

### Standard Benchmark (live)

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|------------------:|------------------------:|---------------------:|-----------------:|----------------------:|------------:|
| Baseline |             6,228 |                  35,995 |                  11% |             0.04 |                     0 |           0 |
| Advanced |            12,646 |                  74,497 |                 100% |             0.89 |                   769 |          14 |

So với Baseline, Advanced tốn thêm **+103% agent tokens** và **+107% prompt tokens**, recall **tăng 89 điểm %**.

### Long-Context Stress Benchmark (live)

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|------------------:|------------------------:|---------------------:|-----------------:|----------------------:|------------:|
| Baseline |             5,451 |                  45,030 |                   0% |             0.07 |                     0 |           0 |
| Advanced |             6,578 |                  20,354 |                 100% |             0.83 |                   544 |           7 |

So với Baseline, Advanced tốn thêm **+21% agent tokens** nhưng **giảm 54.8% prompt tokens**, và recall **tăng 100 điểm %**.

Với token thật, câu chuyện của bản offline vẫn đúng (offline: +131% và −42%): hội thoại ngắn thì Advanced đắt hơn, hội thoại dài thì compact giúp Advanced rẻ hơn hẳn.

## 4. Phân tích

### Vì sao Advanced có recall tốt hơn Baseline?

Mọi câu hỏi recall đều được hỏi ở **thread mới**. Baseline bắt đầu thread mới với session rỗng, nên recall bằng 0%. Advanced đọc `User.md`, file này nằm trên đĩa nên vẫn còn kể cả khi tạo lại instance agent.

Recall đạt 100% còn nhờ `User.md` lưu **fact đã được giải quyết xung đột** chứ không lưu lịch sử thô. Ví dụ: Đà Nẵng → Huế (conv-03), backend → MLOps (conv-06), và Huế → Đà Nẵng (stress). Nhờ vậy câu trả lời luôn lấy bản đính chính mới nhất.

Ở chế độ live, Baseline không chỉ quên mà còn **bịa**:

- được hỏi "Bạn biết DũngCT là ai không?", nó trả lời "DũngCT là một YouTuber nổi tiếng ở Việt Nam";
- với câu "Nếu ai đó nhắc Huế, Hà Nội hay product manager…", nó đoán "có thể bạn đang sống ở Hà Nội và làm product manager", tức là chọn đúng thông tin nhiễu.

Phần recall 11% của Baseline ở live chủ yếu đến từ việc nó lặp lại chữ có sẵn trong câu hỏi (ví dụ "Huế", "DũngCT"), không phải từ trí nhớ. Có persistent memory thì LLM có nguồn sự thật để dựa vào thay vì đoán.

### Vì sao Advanced tốn hơn ở hội thoại ngắn?

Thread trong Standard chỉ dài khoảng 10 lượt ngắn (dưới 400 token), nên **không bao giờ chạm ngưỡng compact** (Compactions = 0). Khi đó Advanced chỉ có thêm chi phí mà chưa có khoản tiết kiệm nào bù lại:

- `User.md` (~200 token) và system prompt dài hơn được inject vào **mọi** lượt, nên prompt tokens tăng 130%;
- mỗi lần ghi memory là một lần gọi tool (`upsert_fact`), nên agent tokens tăng 2.5%.

Với hội thoại ngắn, lịch sử đầy đủ vốn đã rẻ, còn memory dài hạn là một khoản trả trước. Khoản này chỉ có lợi khi người dùng quay lại ở phiên sau.


### Vì sao compact giúp Advanced có lợi thế ở hội thoại dài?

Compact giới hạn kích thước **ngữ cảnh mỗi lượt**: summary (tối đa 8 bullet) cộng 4 message gần nhất cộng `User.md`. Kích thước này gần như không phụ thuộc độ dài hội thoại, còn Baseline thì phải mang theo toàn bộ lịch sử.

Compact chủ yếu tối ưu **`prompt tokens processed`**. Nó gần như không đổi `agent tokens only`, vì số token người dùng và agent nói ra vẫn vậy. Agent tokens thậm chí tăng nhẹ (+7.6%) vì có thêm thao tác ghi memory.

Compact không làm mất fact quan trọng, vì fact đã được rút vào `User.md` **trước khi** message bị nén. Đó là lý do recall vẫn 100% ở mọi ngưỡng. Thứ bị mất là chi tiết hội thoại, ví dụ con số cụ thể của tin WMO hay X-59. Summary heuristic chỉ giữ câu đầu của mỗi lượt. Nếu cần giữ abstraction (readiness, externality, uncertainty, efficiency) thì phải dùng summary bằng LLM, như `SummarizationMiddleware` ở chế độ live.

### Memory file tăng trưởng thế nào và có rủi ro gì?

`User.md` tăng 769 byte sau 10 phiên và 544 byte sau stress test. File tăng theo **số loại fact** chứ không theo số lượt chat, vì:

- field đơn trị bị ghi đè;
- field đa trị (`response_style`, `interests`) được khử trùng lặp và giới hạn 6 giá trị;
- log `Corrections` giới hạn 5 dòng.

Không có các giới hạn này thì file sẽ phình tuyến tính, và vì file được inject vào **mọi** prompt nên mỗi byte thừa bị trả tiền ở mọi lượt.

Một số rủi ro còn lại:

- **Lưu sai fact.** Một fact sai trong `User.md` sẽ làm hỏng mọi phiên sau, nên nó nguy hiểm hơn nhiều so với quên. Ở live, việc để LLM tự ghi memory gây ra đúng lỗi này nhiều lần.
- **Preference cộng dồn.** Ví dụ `response_style` gom 6 tag. Nếu người dùng đổi ý (kiểu "đừng dùng bullet nữa") thì hệ thống hiện chưa xóa tag.
- **Quyền riêng tư.** File chứa thông tin cá nhân dạng plain text. Production cần cơ chế cho người dùng xem và xóa memory.

## 5. Bonus

| Bonus | Cách làm | Giải quyết vấn đề gì | Rủi ro / chi phí |
|---|---|---|---|
| **Confidence threshold** | Mỗi fact có base confidence theo field. Câu bắt đầu bằng "Nếu", hoặc chứa "hay là", "đùa", "giả sử", "ví dụ cũ" bị trừ 0.45. Câu chứa "đính chính", "hiện tại", "thực ra" được cộng 0.05. Chỉ ghi khi đạt ≥ 0.6. Tool live `save_user_fact` cũng áp ngưỡng này. | Chặn câu đùa "chuyển sang product manager" (confidence 0.40, bị loại) và câu giả định "nếu sau này mình nhắc Đà Nẵng…" | Ngưỡng quá cao sẽ bỏ sót fact thật. Ví dụ "mình đang ở Huế … nếu cần" từng bị trừ điểm oan |
| **Conflict handling** | Field đơn trị: giá trị mới ghi đè giá trị cũ, giá trị cũ chuyển sang `## Corrections` (giới hạn 5 dòng). Mục `Profile` không bao giờ chứa đồng thời fact cũ và mới. Câu phủ định ("không còn làm backend…", "chứ không còn ở Đà Nẵng") không sinh fact. | Đúng các case Đà Nẵng→Huế, backend→MLOps, Huế→Đà Nẵng. Câu trả lời không nhắc fact cũ | Recency luôn thắng: một câu mới nhưng sai vẫn ghi đè câu cũ đúng nếu vượt ngưỡng. |
| **Entity extraction có cấu trúc** | 8 field (`name`, `location`, `profession`, `favorite_drink`, `favorite_food`, `pet`, `response_style`, `interests`). Nơi ở lấy từ danh sách thành phố, nghề theo mẫu "<x> engineer/manager…", style được chuẩn hóa thành tag (`ngắn gọn`, `3 bullet`, `nhấn trade-off`…). | File gọn, dễ đọc, dễ trả lời đúng field được hỏi.  | Câu diễn đạt lạ sẽ bị bỏ sót. |
| **Guardrail cho memory do LLM ghi** (`validate_llm_fact`) | Tool `save_user_fact` chỉ nhận giá trị: (1) trích nguyên văn từ tin nhắn hiện tại; (2) không nằm sau từ phủ định; (3) ≤ 6 từ; (4) cho field đơn trị **đang trống**, chưa được extractor xử lý trong lượt đó. Field đa trị và mọi đính chính chỉ do extractor ghi. | Chặn các lỗi thật ở live: ghi đè Huế bằng "Đà Nẵng" từ câu phủ định, `favorite_food: đi biển Mỹ Khê chụp ảnh`, `interests` bị nhét câu dài. Recall live từ 89% lên 100% (test `test_llm_tool_writes_are_guarded`). | LLM gần như mất quyền sửa memory, nên fact mà extractor không nhận ra (câu diễn đạt lạ) sẽ không được đính chính. Benchmark cho thấy với dữ liệu này, tắt hẳn tool rẻ hơn mà không mất recall. |
| **Không lưu khi người dùng hỏi** | Bỏ qua câu hỏi (dấu `?`, "gì", "ở đâu", "là ai"…) và câu yêu cầu nhắc lại ("Nhắc lại giúp…", "Tóm tắt…"). Ngoài ra loại mọi giá trị là từ để hỏi (`gì`, `nào`, `đâu`, `ai`; phân biệt hoa/thường nên `AI` không bị loại). | Sửa lỗi `pet: gì` (test `test_questions_jokes_and_noise_are_not_saved`). | Câu khẳng định có chữ "gì" (ví dụ "không có gì thay đổi") sẽ bị bỏ qua. |