Quy tắc lập trình cho DominicBE

1. Ngữ cảnh dự án
- Là dự án BackendAI sử dụng FastAPI
- Domain chính: Xử lí ngôn ngữ tự nhiên, RAG

2. Quy tắc chung
- Sử dụng duy nhất môi trường Python của repo: `.venv`
- Khi chạy backend, migration, script smoke test hoặc debug, luôn gọi qua `.venv/Scripts/python.exe` trên Windows hoặc `.venv/bin/python` trên Linux
- Code phải rõ ràng, dễ hiểu, dễ bảo trì
- Tại mỗi phần, hãy chỉ ra dự án hiện tại đã làm được gì, chưa làm được gì, và cần làm gì tiếp theo
- Tập trung vào việc hoàn thiện các tính năng cơ bản trước khi mở rộng thêm
- Luôn viết test và kiểm thử trong terminal cho các tính năng đã triển khai
- Đảm bảo code có thể chạy được và không bị lỗi trước khi chuyển sang phần tiếp theo
- Sử dụng các thư viện ở phiên bản mới nhất, tránh dùng các thư viện đã lỗi thời
- Khi có lỗi, hãy debug và sửa lỗi trước khi tiếp tục phát triển thêm tính năng
- Luôn giữ cho codebase sạch sẽ, tránh để lại code thừa hoặc không sử dụng
- Khi thêm tính năng mới, hãy đảm bảo nó được tích hợp tốt với các phần đã có
- Luôn cập nhật README.md và tài liệu liên quan khi có thay đổi về API hoặc cách sử dụng
- Khi thêm mới các nút, component nói chung trên UI, cần đảm bảo tính tương thích với các nút, component khác, đi theo phong cách của UI hiện tại