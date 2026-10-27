# Đưa Knee OA Observatory lên web bằng trình duyệt

Gói có sẵn giao diện dashboard, thống kê tổng hợp, mô hình demo và thư viện cố định. Bạn không cần chạy lệnh để chuẩn bị gói này. Link cố định chỉ có sau khi deploy trên Streamlit Community Cloud.

## 1. Giải nén

Giải nén ZIP vào một thư mục riêng. Không cần chép vào dự án Windows đang chạy. Bên trong phải thấy app.py, dashboard_core.py, requirements.txt, dashboard_data.json, results và .streamlit.

## 2. Đưa lên GitHub

- Vào https://github.com/new và đăng nhập hoặc tạo tài khoản.
- Tạo repository tên knee-oa-dashboard. Chọn chế độ hiển thị phù hợp với việc chia sẻ mã nguồn của bạn.
- Chọn Upload files hoặc uploading an existing file.
- Mở thư mục đã giải nén, chọn các file và thư mục BÊN TRONG rồi kéo vào trang upload. Giữ nguyên results và .streamlit.
- Không upload chỉ file ZIP. app.py phải nằm ở gốc repository.
- Commit changes để lưu.

## 3. Triển khai

- Mở https://share.streamlit.io/ và liên kết tài khoản GitHub.
- Chọn Create app, rồi chọn repository vừa tạo.
- Branch: main. Main file path: app.py.
- Trong Advanced settings, chọn Python 3.12. Gói này được chuẩn bị và kiểm tra với Python 3.12, không cần theo Python 3.14 trên máy Windows của bạn.
- Bấm Deploy. Đợi cài thư viện và khởi động. Khi thành công sẽ có địa chỉ thuộc streamlit.app.
- Kiểm tra quyền xem app trong phần chia sẻ nếu muốn người khác truy cập mà không có quyền vào repository.

## 4. Sau khi triển khai

Bạn có thể đóng VS Code, tắt laptop và mở link trên thiết bị khác. Máy chủ cloud chạy ứng dụng thay máy cá nhân. Miễn phí Community Cloud có thể ngủ sau 12 giờ không có người truy cập; người xem có thể cần nhấn nút đánh thức rồi chờ. Nếu yêu cầu ứng dụng luôn chạy sẵn cần chọn dịch vụ hosting chạy liên tục.

## Nguồn và mô hình trong gói

- Giao diện và mô hình thuộc bản demo thoái hóa khớp gối của dự án.
- Mô hình trong gói là bản tái hiện đã kiểm tra từ cùng bộ dữ liệu và cấu hình Logistic Regression của dự án, bằng scikit-learn 1.8.0; không phải file nhị phân lấy trực tiếp từ máy Windows. Các chỉ số test tái hiện: Accuracy 0.6200, Recall lớp 1 0.5481, ROC-AUC 0.6403. Xem comparison.json và DEPLOY_INFO.json.
- Chỉ có thống kê tổng hợp từ MatchThem dẫn xuất OAI có chỉnh sửa, không có CSV hồ sơ gốc. Gói không phải dữ liệu OAI gốc.
- Nhóm điểm thấp / trung gian / cao là tương đối theo tập train, không phải mức độ bệnh hoặc ngưỡng nguy cơ y khoa.
- Hướng dẫn theo triệu chứng độc lập với ML. Nguồn y khoa và nguồn dữ liệu nằm trong tab Phương pháp & nguồn.

## Kiểm tra sau khi deploy

- Tab Tổng quan dữ liệu có 2.585 hồ sơ ban đầu, 2.392 hồ sơ có nhãn.
- Thử tuổi 68, BMI 28.1, hút thuốc Không, loãng xương Không; điểm mô hình khoảng 0.516, nhãn 1, nhóm điểm trung gian. Đây chỉ là ví dụ nhập liệu.
- Không diễn giải nhãn hoặc điểm thành chẩn đoán hay nguy cơ mắc bệnh sau 48 tháng.

Tài liệu triển khai: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
