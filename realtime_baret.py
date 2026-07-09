import cv2
from ultralytics import YOLO

if __name__ == '__main__':
    print("GERÇEK ZAMANLI BARET TAKİP SİSTEMİ BAŞLATILIYOR")
    model = YOLO('runs/detect/train-3/weights/best.pt')
    cap = cv2.VideoCapture(0)

    print("🟢 Kamera aktif! Kapatmak için klavyeden 'q' tuşuna basın.")

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            print(" Kameradan görüntü alınamadı.")
            break

        results = model.predict(source=frame, stream=True, device=0, imgsz=640, conf=0.65, iou=0.45 )  # 0: head, 1: head_with_helmet
        head_count = 0
        helmet_count = 0
        class_names = model.names

        for result in results:
            annotated_frame = result.plot()

            for box in result.boxes:
                cls_id = int(box.cls[0])
                name = class_names[cls_id]

                if name == 'head':
                    head_count += 1
                elif name == 'head_with_helmet':
                    helmet_count += 1

      
        cv2.putText(annotated_frame, f"Baretsiz (head): {head_count}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(annotated_frame, f"Baretli (helmet): {helmet_count}", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

       
        cv2.imshow(" Canlı Baret Takip Sistemi", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(" Sistem kapatıldı.")
