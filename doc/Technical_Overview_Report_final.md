# TECHNICAL OVERVIEW REPORT

## Vietnamese Social Listening NLP Model

### Brand-Conditioned Sentiment Classification & Single-label Topic Classification

**Ngày cập nhật:** 06/11/2026

**Phạm vi:** Vietnamese NLP – Social Listening – Transformer Models – Multi-task Learning

---

## EXECUTIVE SUMMARY

Dự án xây dựng NLP Engine cho Social Listening tiếng Việt với hai đầu ra: sentiment theo từng cặp `(comment_id, brand)` và topic chính theo từng `comment_id`.
| Task | Input chính | Output | Đơn vị dự đoán |
|---|---|---|---|
| Brand-conditioned Sentiment | Brand + Target (+ Context) | 1 trong 3 lớp | `(comment_id, brand)` |
| Topic Classification | Target (+ Context khi cần) | 1 trong 13 topic | `comment_id` |

**Sentiment:** `0 = Tiêu cực`, `1 = Tích cực`, `2 = Trung lập`.

**Topic:** chọn đúng một topic chính trong T01–T13 cho mỗi comment.

## PART I. PROBLEM FORMULATION

### 1. Cấu trúc dữ liệu

Schema nguồn gồm `comment_id`, `thread_id`, `brand`, `info`, `sentiment`, `topic`. Một comment có thể xuất hiện ở nhiều dòng khi nhắc đến nhiều thương hiệu.
| Field | Cấp độ | Vai trò |
|---|---|---|
| `comment_id` | Comment | Định danh bình luận |
| `thread_id` | Conversation | Group split, chống leakage |
| `brand` | Comment–brand | Input cho Sentiment |
| `info` | Comment | Chứa target và context |
| `sentiment` | Comment–brand | Nhãn 0/1/2 |
| `topic` | Comment | Một topic chính T01–T13 |

Ví dụ minh họa cấu trúc: một comment nhắc hai brand vẫn chỉ có một topic. Sentiment dưới đây chỉ minh họa hai record riêng biệt, không phải dữ liệu thực:
| comment_id | brand | sentiment | topic |
|---|---|---:|---|
| C001 | VinFast | — | T01 – Pin & Sạc |
| C001 | Toyota | — | T01 – Pin & Sạc |

Khi tạo dữ liệu Topic, phải deduplicate theo `comment_id`; không nhân bản một comment theo số brand. Nhãn Sentiment được quản lý riêng theo từng cặp `(comment_id, brand)`.

### 2. Tách Target và Context

Trường `info` có cấu trúc gồm `thread_info`, `context` và `target`. Pipeline cần parse thành các trường riêng:
| Thành phần | Mục đích sử dụng |
|---|---|
| `target` | Nội dung chính để dự đoán |
| `context` | Thông tin hỗ trợ diễn giải khi target ngắn hoặc phụ thuộc ngữ cảnh |
| `thread_info` | Metadata; chỉ đưa vào model khi có lý do thực nghiệm rõ ràng |

Topic sử dụng Target và bổ sung Context khi Context tồn tại. Sentiment sử dụng Brand + Target và bổ sung Context khi Context tồn tại. Quy tắc này được áp dụng thống nhất cho các model cùng task; Context không thay thế Target.

### 3. Định nghĩa bài toán

**Sentiment Classification:** phân loại đơn nhãn 3 lớp cho mỗi `(comment_id, brand)`. Đầu ra dùng Softmax và Cross Entropy.

**Topic Classification:** phân loại đơn nhãn 13 lớp cho mỗi `comment_id`. Mỗi comment chỉ có một topic chính; đầu ra dùng Softmax và Cross Entropy.
| Thuộc tính | Sentiment | Topic |
|---|---|---|
| Đơn vị dự đoán | `(comment_id, brand)` | `comment_id` |
| Số lớp | 3 | 13 |
| Output layer | 3 logits + Softmax | 13 logits + Softmax |
| Loss cơ sở | Cross Entropy | Cross Entropy |
| Metric chính | Macro-F1 | Macro-F1 |

Hai task có thể chia sẻ encoder, nhưng phải tính loss và metric theo đúng đơn vị dự đoán của từng task.

## PART II. TAXONOMY & LABEL SCHEMA

### 4. Topic taxonomy

Sử dụng 13 topic cố định. Mỗi `comment_id` nhận **một nhãn chính duy nhất**.
| ID | Topic |
|---|---|
| T01 | Pin & Sạc | Pin, sạc hoặc đổi pin nói chung, không tập trung vào hạ tầng hay chi phí |
| T02 | Chất lượng hạ tầng Sạc/ Đổi Pin | Độ phủ, khả năng hoạt động và trải nghiệm tại trạm/trụ sạc hoặc điểm đổi pin |
| T03 | Chính sách hậu mãi (bảo hành, bảo dưỡng, sửa chữa) | Quyền lợi, điều kiện và chính sách hậu mãi |
| T04 | Quy trình đặt hẹn dịch vụ | Đặt lịch, xác nhận, thay đổi hoặc hủy lịch dịch vụ |
| T05 | Hệ thống Xưởng Dịch vụ (XDV) | Mạng lưới, địa điểm và cơ sở vật chất xưởng |
| T06 | Chất lượng dịch vụ tại XDV | Chất lượng thực hiện và kết quả dịch vụ tại xưởng |
| T07 | Thái độ và chuyên môn của nhân sự XDV | Thái độ, kỹ năng và năng lực nhân viên xưởng |
| T08 | Chi phí sửa chữa, bảo dưỡng, thay thế phụ tùng | Giá và khoản phí sửa chữa/bảo dưỡng/phụ tùng |
| T09 | Chính sách cứu hộ, hỗ trợ khẩn cấp | Điều kiện và quyền lợi cứu hộ/hỗ trợ khẩn cấp |
| T10 | Tổng đài CSKH | Hoạt động và trải nghiệm liên hệ tổng đài chăm sóc khách hàng |
| T11 | Dịch vụ sửa chữa tại nhà | Hoạt động sửa chữa/bảo dưỡng được thực hiện tại nhà |
| T12 | Đề cập chung | Nội dung về xe/thương hiệu không thuộc topic chuyên biệt khác |
| T13 | Chi phí & chính sách sạc | Giá, phí, ưu đãi và chính sách thanh toán cho sạc/đổi pin |

## PART III. PREPROCESSING METHODOLOGY

Khác với các bài toán NLP truyền thống (như phân loại văn bản hành chính hay tin tức), văn bản mạng xã hội (Social Media) có tính phi hình thức cao, chứa nhiều emoji, teencode, từ mượn và sắc thái cảm xúc đặc thù. Do đó, hệ thống áp dụng triết lý **Conservative Preprocessing** (tiền xử lý bảo toàn thông tin), kiên quyết tránh cách tiếp cận **Destructive Cleaning** (làm sạch phá hủy thông tin).

### 5. Triết lý Tiền Xử Lý Mạng Xã Hội (Conservative vs. Destructive Cleaning)

#### 5.1. Phân tích nguyên tắc bảo toàn Thông tin Mạng Xã Hội (Conservative vs. Destructive)

Nhiều quy trình tiền xử lý cũ trên mạng áp dụng các quy tắc lọc bừa bãi (như xóa emoji, xóa dấu câu, hạ chữ thường toàn bộ, hoặc xóa dấu tiếng Việt). Phân tích kỹ thuật đối với hệ thống như sau:

| Thao tác xử lý | Đánh giá & Rủi ro | Giải pháp áp dụng trong dự án |
|---|---|---|
| **Xử lý Emoji** | ❌ **Xóa emoji làm giảm sút nghiêm trọng Macro-F1 Sentiment.** Trên mạng xã hội, emoji (như 😍, 👍, 🔥 vs 😡, 🤮, 💔, 😭) là vật mang cảm xúc chính. Nghiên cứu *ViGoEmotions (EACL 2026)* chỉ ra rằng giữ nguyên emoji đem lại kết quả tốt nhất cho ViSoBERT. | **Giữ nguyên 100% Emoji.** Không chuyển đổi sang text và không dùng regex để loại bỏ emoji. |
| **Chữ hoa / Chữ thường (Case)** | ❌ **Hạ chữ thường toàn bộ (`lower()`) làm mất tín hiệu thực thể.** Tên thương hiệu (`VinFast`, `BYD`, `Toyota`) và các từ viết tắt kỹ thuật ngành ô tô/xe điện (`XDV` - Xưởng Dịch Vụ, `CSKH`, `PIN`) cần giữ nguyên case để Transformer nhận diện chính xác. Ngoài ra, viết hoa cả từ (`QUÁ TỆ`) thể hiện độ gay gắt của cảm xúc. | **Giữ nguyên Case (Cased representation).** Các pretrained Transformer (ViSoBERT, PhoBERT) đều là cased models. |
| **Dấu câu (Punctuation)** | ❌ **Xóa toàn bộ dấu câu làm mất ngữ khí.** Dấu `!`, `?`, `?!`, `...` biểu thị sự bức xúc, hoài nghi hoặc mỉa mai (ví dụ: *"Hãng bảo hành tốt thật đấy?!"* vs *"Hãng bảo hành tốt thật đấy"*). Dấu câu còn giúp Transformer phân định ranh giới mệnh đề. | **Giữ nguyên dấu câu tự nhiên.** Chỉ loại bỏ các ký tự điều khiển (control characters) và chuẩn hóa khoảng trắng thừa. |
| **Rút gọn ký tự lặp** | ⚠️ **Dễ gây hỏng từ vựng tiếng Việt.** Các regex rút gọn thông thường (như `([a-z]+)\1+` -> `\1`) sẽ làm sai các từ như `Facebook` -> `Facebok`, `boong` -> `bong`, `buffet` -> `bufet`, và không hỗ trợ nguyên âm tiếng Việt (`áaaa`, `oooo`). | **Không áp dụng regex rút gọn mù quáng.** Mô hình Transformer với subword tokenization hoàn toàn có thể tự học cách xử lý các biến thể kéo dài từ. |
| **Bỏ dấu tiếng Việt (`unidecode`)** | ❌ **Tuyệt đối cấm kỵ.** Tiếng Việt không dấu làm thay đổi hoàn toàn ngữ nghĩa (`xe chạy pin` vs `xe chay pin`, `mất tiền` vs `mat tien`). | **Luôn duy trì đầy đủ dấu tiếng Việt chuẩn.** |

### 6. Quy trình Chuẩn hóa Chuỗi (Text Normalization)

Mỗi đoạn văn bản trước khi đưa vào mô hình đều trải qua các bước chuẩn hóa chuẩn mực theo quy trình tiền xử lý văn bản:
1. **Unicode NFC Normalization:** Đồng bộ toàn bộ văn bản về chuẩn dựng sẵn UTF-8 NFC (`unicodedata.normalize("NFC", text)`), giải quyết triệt để lỗi xung đột giữa font tổ hợp và dựng sẵn do các bộ gõ tiếng Việt (Unikey, EVKey) gây ra.
2. **Dọn dẹp khoảng trắng:** Thay thế khoảng trắng không ngắt (`\u00a0`) thành khoảng trắng thường, gộp các khoảng trắng liên tiếp (`\s+`) thành một khoảng trắng đơn và loại bỏ khoảng trắng thừa ở hai đầu chuỗi.
3. **Lọc chỉ thị thiếu:** Bất kỳ giá trị nào mang nhãn rác hoặc thiếu (`(không có)`, `không có`, `none`, `nan`, `null`, `n/a`) đều được quy về chuỗi rỗng `""`.

### 7. Chiến lược Phân đoạn Từ (Word Segmentation): PyVi vs. Underthesea vs. Subword BPE

Tách từ ghép tiếng Việt (ví dụ `xe điện` -> `xe_điện`, `trạm sạc` -> `trạm_sạc`) là bước quan trọng, tuy nhiên việc áp dụng phải phụ thuộc chính xác vào cấu trúc của từng mô hình:

- **Đối với ViSoBERT / XLM-R / Decoder LLM (Qwen, v.v.):**
  - **Không dùng Word Segmentation.** Các mô hình này được huấn luyện với tokenizer dạng subword (Byte-level BPE / SentencePiece) trên văn bản tự nhiên **raw text (không có dấu gạch dưới)**.
  - Nếu đưa chuỗi đã gạch dưới vào (như từ `underthesea.word_tokenize`), tokenizer của ViSoBERT sẽ bị tách vỡ thành các token vụn (`xe`, `_`, `điện`), gây hiện tượng OOV (Out-Of-Vocabulary) giả tạo và làm suy giảm độ chính xác.
- **Đối với PhoBERT (`vinai/phobert-base-v2`):**
  - **Bắt buộc phân đoạn từ trước khi đưa vào tokenizer**, vì PhoBERT được huấn luyện trên kho ngữ liệu đã tách từ có dấu nối gạch dưới `_`.
  - Hệ thống tích hợp thư viện **`pyvi`** cho nhiệm vụ phân đoạn từ tiếng Việt (Word Segmentation). `pyvi` được chọn vì có sự tương thích cao nhất với chuẩn từ điển `VnCoreNLP` của nhóm tác giả PhoBERT, tốc độ xử lý nhanh, vận hành ổn định và không đòi hỏi tải thêm checkpoint trọng số bên ngoài.
- **Đối với Baseline TF-IDF + Linear SVM:**
  - TF-IDF truyền thống thường gặp bất lợi do tách rời các từ ghép tiếng Việt thành các âm tiết đơn vô nghĩa.
  - Dự án giải quyết triệt để vấn đề này bằng kiến trúc **`FeatureUnion`**, kết hợp song song:
    - **Word n-grams (1, 2):** Tự động gom các cụm từ ghép liền kề 2 từ (như "trạm sạc", "bảo hành", "cứu hộ").
    - **Char-wb n-grams (3, 5):** N-gram ký tự bên trong ranh giới từ, giúp bắt được cả gốc từ, từ viết tắt và biến thể teencode.

### 8. Quy cách Xây dựng Chuỗi Đầu Vào (Task Input Formatting)

Các chuỗi đầu vào được định dạng nhất quán trên toàn bộ các mô hình theo quy chuẩn cấu trúc sau:

- **Brand-Conditioned Sentiment Input:**
  ```text
  Brand: {brand} [SEP] Target: {target} [SEP] Context: {context}
  ```
  *(Nếu `context` rỗng, phần `[SEP] Context: ...` tự động được lược bỏ).*
- **Topic Classification Input:**
  ```text
  Target: {target} [SEP] Context: {context}
  ```
  *(Nếu `context` rỗng, chuỗi chỉ bao gồm `Target: {target}`).*
- **Decoder Prompt Template:**
  Sử dụng định dạng hội thoại (Chat Template) có phân định vai trò `System` và `User`, nêu rõ danh sách nhãn hợp lệ (`0 = Negative, 1 = Positive, 2 = Neutral` cho Sentiment và `T01–T13` cho Topic) nhằm loại trừ hoàn toàn việc mô hình sinh nhãn ngoài danh mục.
- **Ma trận Khảo sát Biến thể (Input Ablation Matrix):**
  Hệ thống thiết lập sẵn các biến thể input ablation để kiểm chứng mức độ phụ thuộc dữ liệu:
  - Sentiment: `full`, `brand_target`, `target_context`, `target_only`, `brand_only`, `context_only`, `full_masked_brand`, `full_target_marked`.
  - Topic: `target_context`, `target_only`, `context_only`, `target_context_marked`, `target_context_masked_brand`, `target_masked_brand`, `context_masked_brand`.

## PART IV. LITERATURE REVIEW & STATE OF THE ART

### 9. Sentiment Classification – Các hướng nghiên cứu hiện có

Sentiment Classification đã phát triển qua nhiều thế hệ:

1. Lexicon-based và traditional machine learning.
2. CNN, RNN, BiLSTM.
3. Transformer-based contextual encoders.
4. Aspect-Based Sentiment Analysis.
5. Target-Based Sentiment Analysis.
6. Instruction-tuned LLM và domain adaptation.

Đối với dự án, hướng 4 và 5 có liên quan trực tiếp nhất.

#### 9.1. Benchmark và nghiên cứu tiếng Việt
| Công trình | Dataset / Domain | Đóng góp | Giá trị cho dự án |
|---|---|---|---|
| UIT-VSFC | Phản hồi sinh viên | Sentiment và topic classification | Baseline multi-task |
| VLSP 2016 | Sentiment tiếng Việt | Benchmark sentiment truyền thống | Baseline tham chiếu |
| NEU-ESC | Forum giáo dục | Sentiment/topic multi-task | Shared encoder |
| VLSP 2018 ABSA | Restaurant, hotel | Aspect-category sentiment | Target/aspect conditioning |
| ViTASA (2025) | Social media, mobile, restaurant, hotel | Targeted aspect sentiment | Rất gần target sentiment |
| ViGoEmotions (2026) | Vietnamese social media | 27 emotion classes | Social-media encoder benchmark |
| ViTOED (2026) | Vietnamese social media | Target-oriented opinion structures | Structured target sentiment |

##### UIT-VSFC

- Hơn 16.000 phản hồi.
- Ba sentiment classes.
- Bốn nhóm topic.
- Có thể dùng làm tham chiếu cho joint sentiment-topic learning.

Hạn chế: sentiment không condition theo brand.

##### NEU-ESC

- 32.966 mẫu.
- Bốn sentiment classes.
- Mười topic.
- So sánh nhiều Vietnamese pretrained encoders.
- Khảo sát single-task và multi-task.

Paper báo cáo kết quả tốt nhất khoảng 83,7% accuracy sentiment và 79,8% accuracy topic trong thiết lập của nghiên cứu.

Hạn chế: nhãn và đơn vị dự đoán khác dự án hiện tại.

#### 9.2. ViTASA – Benchmark gần nhất cho target sentiment

ViTASA: New benchmark and methods for Vietnamese targeted aspect sentiment analysis for multiple textual domains (2025).

Đặc điểm:

- Hơn 500.000 target-aspect pairs.
- Ba domain: mobile, restaurant, hotel.
- Xây dựng ViTASD dựa trên PhoBERT.
- Đánh giá nhiều baseline BERT-based và zero-shot LLM.

Kết quả Macro-F1 được công bố:
| Domain | ViTASD Macro-F1 |
|---|---:|
| Mobile | 61,77% |
| Restaurant | 41,12% |
| Hotel | 52,64% |

Ý nghĩa:

Target-based sentiment khó hơn sentence-level sentiment vì model phải xác định chính xác thái độ đối với một đối tượng cụ thể.

Liên hệ với dự án:

- Brand đã được cung cấp.
- Không cần extract brand span.
- Không cần extract aspect span ở phiên bản hiện tại.
- Cần xác định sentiment đối với brand tương ứng.

Vì vậy, bài toán hiện tại có thể đơn giản hơn TASA ở khía cạnh extraction, nhưng vẫn phải giải quyết target-conditioned polarity.

#### 9.3. ViGoEmotions – EACL 2026

ViGoEmotions giới thiệu:

- 20.664 bình luận tiếng Việt.
- 27 loại cảm xúc.
- So sánh tám pretrained Transformer models.
- Nghiên cứu preprocessing emoji và lexical normalization.

Kết quả nổi bật:
| Model | Macro-F1 |
|---|---:|
| ViSoBERT | 61,50% |

Nghiên cứu ghi nhận tác động của cách xử lý emoji khác nhau giữa các pretrained model: giữ nguyên emoji có lợi cho ViSoBERT và CafeBERT, trong khi chuyển emoji thành mô tả văn bản có thể có lợi cho một số baseline khác.

Giá trị cho dự án: không nên loại bỏ emoji hàng loạt trong preprocessing. Giữ nguyên emoji làm mặc định; chỉ xem xét thay đổi khi error analysis cho thấy ảnh hưởng rõ rệt.

ViGoEmotions là emotion classification, không phải brand-conditioned sentiment. Không được gọi trực tiếp 61,50% là benchmark của task hiện tại.

#### 9.4. ViTOED – 2026

ViTOED giới thiệu:

- 10.985 comments.
- 21.244 opinion quadruples.
- Source, target, expression, polarity.
- Structured sentiment graph.

Mô hình phải xác định mối quan hệ giữa target và opinion expression.

Giá trị:

Bài toán của dự án không cần trích xuất đầy đủ quadruple, nhưng có thể sử dụng tư tưởng target-aware representation để cải thiện sentiment.

---

### 10. Brand-Conditioned Sentiment – Kiến trúc hiện đại

#### 10.1. BERT-pair / Cross-Encoder

Đây là baseline quan trọng.

Input:

```text
[CLS] Brand: VinFast
[SEP] Target: Nội dung bình luận
[SEP]
```

Model dự đoán sentiment cho brand được chỉ định.

Ưu điểm:

- Brand được đưa trực tiếp vào Transformer attention.
- Học được quan hệ giữa brand và nội dung.
- Dễ triển khai bằng Hugging Face.
- Phù hợp target-dependent sentiment.

Nhược điểm:

- Phải encode lại nội dung cho mỗi brand.
- Chi phí tăng khi một comment có nhiều brand.

#### 10.2. Shared Encoder + Brand Fusion

Cấu trúc:

```text
Target
  |
Shared Encoder
  |
  h_target
        \
         Fusion ---- Sentiment Head
       /
Brand Encoder
  |
  h_brand
```

Biểu diễn fusion:

$$
h_f = [h_t; h_b; |h_t - h_b|; h_t \odot h_b]
$$

Trong đó:

- $h_t$: target representation.
- $h_b$: brand representation.
- $|h_t - h_b|$: absolute difference.
- $h_t \odot h_b$: element-wise interaction.

Ưu điểm:

- Có thể encode target một lần.
- Tận dụng được nhiều brand trong cùng comment.
- Phù hợp hierarchical batching.

Nhược điểm:

- Brand embedding đơn giản có thể chưa đủ để hiểu brand trong câu phức tạp.
- Fusion có thể yếu hơn cross-encoder khi cần tương tác token-level.

#### 10.3. Context-aware Target Attention

Có thể mở rộng bằng:

- Target-aware attention.
- Local-context attention.
- Relative position embedding.
- Dependency-aware attention.

---

### 11. Topic Classification – Literature và State of the Art

Topic của dự án là **single-label multiclass classification** với 13 lớp cố định. Mỗi comment được dự đoán một topic chính; do đó hướng nghiên cứu trực tiếp là Vietnamese text classification, domain adaptation và xử lý mất cân bằng lớp.

#### 11.1. Công trình tham chiếu
| Công trình | Hướng tiếp cận | Giá trị tham khảo |
|---|---|---|
| UIT-VSFC | Vietnamese sentiment/topic classification | Baseline phân loại văn bản tiếng Việt |
| NEU-ESC | Shared encoder và multi-task heads | Tham khảo biểu diễn dùng chung |
| PhoBERT / ViSoBERT | Transformer tiếng Việt | Encoder baseline phù hợp |
| VieJobBERT / VietJobs-37K | Phân loại nghề nghiệp tiếng Việt, multi-label | Benchmark gần về ngôn ngữ; khác formulation |

Các nghiên cứu multi-label không phải formulation của dự án hiện tại; chỉ giữ làm tham chiếu nền nếu cần. Mô hình Topic sử dụng một output head 13 lớp và dự đoán đúng một topic chính.

#### 11.2. Hướng mô hình phù hợp

1. TF-IDF với word/character n-grams và Linear SVM làm baseline cổ điển.
2. Vietnamese pretrained encoder với linear classification head và Softmax.
3. Weighted Cross Entropy nếu phân phối 13 lớp mất cân bằng đáng kể.
4. Multi-task learning chỉ được xem xét sau khi có kết quả single-task đáng tin cậy.

## PART V. TRAINING ARCHITECTURE

### 12. Sentiment Model

#### 12.1. Baseline architecture

```text
Brand + Target (+ Context)
          |
   Vietnamese Encoder
          |
        Pooling
          |
  3-logit Classification Head
          |
        Softmax
          |
 Negative / Positive / Neutral
```

Input chuẩn: `Brand: {brand} [SEP] Target: {target} [SEP] Context: {context}`. Nếu Context rỗng thì bỏ trường Context; không thay đổi quy tắc giữa các encoder cùng task.

#### 12.2. Training objective

Sentiment là phân loại đơn nhãn 3 lớp, dùng Cross Entropy. Chỉ thử Weighted Cross Entropy khi class imbalance thể hiện rõ trên training data; chọn cấu hình bằng validation Macro-F1.

### 13. Topic Model

#### 13.1. Baseline architecture

```text
Target + Context (Context được thêm khi tồn tại)
             |
    Vietnamese Encoder
             |
           Pooling
             |
   13-logit Classification Head
             |
           Softmax
             |
      Argmax: one main topic
```

#### 13.2. Training objective

Topic là phân loại đơn nhãn 13 lớp. Dùng Cross Entropy làm loss cơ sở; có thể dùng class-weighted Cross Entropy nếu phân phối nhãn mất cân bằng. Dự đoán cuối cùng là lớp có xác suất Softmax cao nhất.

Không sử dụng Sigmoid độc lập, BCE, per-label thresholding hoặc multi-label output trong formulation hiện tại.

## PART VI. MULTI-TASK LEARNING

### 14. Hierarchical Multi-task Architecture

Sau khi có single-task baselines, có thể thử một shared encoder với hai nhánh đầu ra:

```text
                 TARGET + CONTEXT
                         |
                   Shared Encoder
                         |
                   Comment Vector
                    /          \
                   /            \
          Topic Classification   Brand Fusion
             13 logits              + Brand
               Softmax                |
           One topic/comment    Sentiment Head
                                     3 logits
                                     Softmax
```

Topic được tính một lần cho mỗi `comment_id`; Sentiment được tính cho từng `(comment_id, brand)`. Khi batching, không nhân bản hoặc tăng trọng số Topic vì một comment xuất hiện ở nhiều dòng brand.

Tổng loss có thể viết:

$$
L = L_{\text{sentiment}} + \lambda L_{\text{topic}}
$$

Trong đó cả hai loss đều là Cross Entropy; `λ` được chọn trên validation. Topic loss tính một lần trên mỗi `comment_id`; Sentiment loss tính trên từng `(comment_id, brand)`.

Sentiment branch dùng brand representation kết hợp với comment representation, ví dụ $h_f = [h_{\text{comment}}; h_{\text{brand}}; |h_{\text{comment}} - h_{\text{brand}}|; h_{\text{comment}} \odot h_{\text{brand}}]$. MTL chỉ được giữ nếu cải thiện hoặc duy trì hợp lý chất lượng của cả hai task; nếu có negative transfer, ưu tiên hai mô hình độc lập.

## PART VII. EXPERIMENT DESIGN

### 15. Core Experiment Plan – 8 thí nghiệm

Giới hạn vòng benchmark chính ở **8 thí nghiệm**. Các thử nghiệm phụ chỉ được mở khi có bằng chứng từ error analysis hoặc kết quả validation.
| ID | Task | Model / phương pháp | Input và thiết lập | Mục đích |
|---|---|---|---|---|
| E01 | Sentiment | TF-IDF + Linear SVM | Brand + Target + Context nếu có | Baseline cổ điển |
| E02 | Sentiment | ViSoBERT | Brand + Target + Context nếu có | Encoder chính |
| E03 | Sentiment | PhoBERT-base-v2 | Cùng input với E02 | Đối chứng encoder |
| E04 | Topic | TF-IDF + Linear SVM | Target + Context nếu có, 13 lớp | Baseline cổ điển |
| E05 | Topic | ViSoBERT | 13 logits, Softmax + CE | Encoder chính |
| E06 | Topic | PhoBERT-base-v2 | Cùng input với E05 | Đối chứng encoder |
| E07 | Joint | Shared ViSoBERT + Topic head + Brand-conditioned Sentiment head | Topic 13 lớp; Sentiment 3 lớp | Đo lợi ích/negative transfer của MTL |
| E08 | Joint challenger | Qwen3.5-4B LoRA SFT | Một model; task prefix riêng, output label ID hợp lệ cho từng task | Đánh giá SLM open-weight |

### 16. Execution Protocol

- Một train/validation/test split cố định, group theo `thread_id`; không để cùng thread xuất hiện ở nhiều tập.
- Dùng cùng label mapping, preprocessing và định nghĩa input giữa các model cùng task.
- E08 dùng output dạng label ID: Sentiment `0/1/2`; Topic `T01–T13`. Chỉ tính loss trên phần output cần dự đoán; inference dùng decoding xác định và kiểm tra output hợp lệ.
- Chọn checkpoint và hyperparameters bằng validation; chỉ đánh giá test sau khi chốt cấu hình.
- Chạy tối thiểu 3 random seeds cho các cấu hình encoder chính; báo cáo mean ± standard deviation.
- Ghi nhận Macro-F1, per-class/per-topic F1, training time, inference latency và throughput.
- Không chạy ensemble, label attention, label correlation hoặc toàn bộ SLM shortlist trong vòng đầu. Chỉ mở rộng khi kết quả 8 thí nghiệm cho thấy một khoảng trống cụ thể.

## PART VIII. EVALUATION & BENCHMARKING

### 17. Metrics

Sentiment – đơn vị `(comment_id, brand)`
| Metric | Vai trò |
|---|---|
| Macro-F1 | Primary |
| Per-class F1 | Theo dõi Negative / Positive / Neutral |
| Accuracy | Tổng thể |
| Confusion Matrix | Phân tích nhầm lẫn |

Topic – đơn vị `comment_id`
| Metric | Vai trò |
|---|---|
| Macro-F1 | Primary, trung bình F1 của 13 lớp |
| Per-topic F1 | Theo dõi từng topic |
| Accuracy | Tỷ lệ dự đoán đúng một topic |
| Weighted-F1 | Bổ sung theo tần suất lớp |
| Confusion Matrix | Phân tích nhầm lẫn giữa các topic |

Với multiclass single-label, Micro-F1 tương đương Accuracy nên không cần báo cáo cả hai. Không sử dụng Samples-F1, Exact Match Ratio hoặc Hamming Loss vì đây không còn là bài toán multi-label.

### 18. Model selection criteria

Chọn model theo hai nhóm tiêu chí, không chỉ dựa vào một điểm số:
| Nhóm | Tiêu chí |
|---|---|
| Chất lượng | Macro-F1, per-class/per-topic F1, confusion matrix |
| Vận hành | P95 latency, throughput, peak RAM/VRAM, model size và độ ổn định |
| Khả năng sử dụng | License/checkpoint terms, khả năng triển khai nội bộ và bảo trì |

Research/benchmark winner là model có chất lượng tốt nhất trong thiết lập đánh giá. Product candidate còn phải đáp ứng yêu cầu vận hành và điều kiện giấy phép. Đo latency và memory trên cùng phần cứng, cùng batch size và cùng quy trình inference.

## PART IX. RISK ANALYSIS

### 19. Rủi ro chính
| Rủi ro | Tác động | Kiểm soát |
|---|---|---|
| Mất cân bằng 13 topic | Macro-F1 thấp ở lớp hiếm | Theo dõi per-topic F1; thử class-weighted CE |
| Thread leakage | Test score bị thổi phồng | Group split theo `thread_id` |
| Context leakage | Mô hình phụ thuộc quá mức vào context | Kiểm tra bằng error analysis và đối chiếu mẫu; không mặc định mở thêm cấu hình benchmark |
| Brand memorization | Sentiment không thực sự phụ thuộc target | Brand-swap/error analysis |
| Negative transfer | MTL kém hơn mô hình riêng | So sánh E07 với single-task baselines |
| Domain drift | Chất lượng giảm theo thời gian | Theo dõi phân phối dữ liệu và metric |

## CONCLUSION – TECHNICAL TAKEAWAYS

Báo cáo xác định một NLP Engine gồm hai nhiệm vụ: (1) Sentiment 3 lớp theo từng cặp `(comment_id, brand)` và (2) Topic single-label 13 lớp theo từng `comment_id`. Phạm vi benchmark vòng đầu gồm tám cấu hình: baseline TF-IDF + Linear SVM, Vietnamese encoders, một cấu hình Multi-task Learning và một SLM Qwen3.5-4B LoRA SFT.

Kết quả được so sánh bằng Macro-F1 và các chỉ số theo lớp, đồng thời xem xét latency, throughput, memory và license trước khi lựa chọn hướng triển khai. EDA sẽ được bổ sung bằng số liệu tính trực tiếp từ dataset; báo cáo không giả định trước phân phối hoặc quy mô dữ liệu.

---

## TÀI LIỆU THAM KHẢO

### Vietnamese Sentiment & Social Media

1. UIT-VSFC – Vietnamese Students' Feedback Corpus.
2. NEU-ESC – Vietnamese Educational Social Media Corpus.
3. ViSoBERT – Vietnamese Social Media Pretrained Language Model, EMNLP 2023.
4. ViGoEmotions – A Benchmark Dataset for Fine-grained Emotion Detection on Vietnamese Texts, EACL 2026.
5. ViTOED – A Dataset for Target-Oriented Emotion Detection on Vietnamese Social Media Texts, MAPR 2026.

### Targeted / Aspect-Based Sentiment

6. VLSP 2018 Aspect-Based Sentiment Analysis.
7. ViTASA – New Benchmark and Methods for Vietnamese Targeted Aspect Sentiment Analysis, Computer Speech & Language, 2025.
8. BERT-pair for Aspect-Based Sentiment Classification.
9. ViGSA – Multi-Task Aspect-Based Sentiment Analysis, 2025 preprint.

### Topic Classification và Vietnamese NLP

10. Vietnamese text classification benchmarks: UIT-VSFC và NEU-ESC.
11. VieJobBERT and VietJobs-37K, IEEE Access 2026 (tham khảo thêm; formulation gốc multi-label).

### Vietnamese Pretrained Models

12. PhoBERT.
13. ViSoBERT.
14. XLM-R (tham chiếu bổ sung khi cần xử lý code-switching).

### Open-weight SLM/LLM candidates

15. Qwen Team. **Qwen3 Technical Report** and Qwen3 model repository. [https://github.com/QwenLM/Qwen3](https://github.com/QwenLM/Qwen3)
16. VinAI Research. **PhoGPT: Generative Pre-training for Vietnamese**. [https://arxiv.org/abs/2311.02945](https://arxiv.org/abs/2311.02945)
17. Microsoft. **Phi-4-mini-instruct Model Card**. [https://huggingface.co/microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)
18. Qwen Team. **Qwen3.5-4B Model Card**. [https://huggingface.co/Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
19. Sea AI Lab. **Sailor2 Model Repository**. [https://huggingface.co/sail](https://huggingface.co/sail)
20. AI Singapore. **SEA-HELM Leaderboard – Vietnamese Performance**. [https://leaderboard.sea-lion.ai/detailed/VI](https://leaderboard.sea-lion.ai/detailed/VI)

---
