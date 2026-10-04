## 1. Kết quả benchmark (offline, cấu hình mặc định)

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

## 1b. Kết quả chế độ live (OpenAI `gpt-4o-mini`, token thật)

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

## 2. Phân tích

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