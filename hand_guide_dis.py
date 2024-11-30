import cv2
import mediapipe as mp
import numpy as np

class HandGuidanceSystem:
    def __init__(self, num_segments=20, line_start=(100, 100), line_end=(400, 400)):
        # Mediapipeのセットアップ
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=1  # 一つの手に制限して処理を効率化
        )
        self.mp_draw = mp.solutions.drawing_utils

        # ガイドラインの設定
        self.num_segments = num_segments
        self.line_start = line_start
        self.line_end = line_end
        self.segments = self._create_segments()
        self.visited_segments = [False] * num_segments
        self.current_segment_index = 0

        # 色の定義
        self.COLORS = {
            'unvisited': (0, 255, 0),    # 緑
            'current': (0, 255, 255),     # 黄
            'visited': (0, 0, 255),       # 赤
            'completed': (255, 0, 0)      # 青
        }

    def _create_segments(self):
        segments = []
        for i in range(self.num_segments):
            start = (
                int(self.line_start[0] + i * (self.line_end[0] - self.line_start[0]) / self.num_segments),
                int(self.line_start[1] + i * (self.line_end[1] - self.line_start[1]) / self.num_segments)
            )
            end = (
                int(self.line_start[0] + (i + 1) * (self.line_end[0] - self.line_start[0]) / self.num_segments),
                int(self.line_start[1] + (i + 1) * (self.line_end[1] - self.line_start[1]) / self.num_segments)
            )
            segments.append((start, end))
        return segments

    def _is_near_line(self, point, line_start, line_end, threshold=10):
        line_vec = np.array(line_end) - np.array(line_start)
        point_vec = np.array(point) - np.array(line_start)

        line_len = np.linalg.norm(line_vec)
        if line_len == 0:
            return False

        line_unit_vec = line_vec / line_len
        proj = np.dot(point_vec, line_unit_vec)

        # セグメント内に収まっているか確認
        if proj < 0 or proj > line_len:
            return False

        closest_point = np.array(line_start) + proj * line_unit_vec
        distance = np.linalg.norm(np.array(point) - closest_point)

        return distance < threshold

    def process_frame(self, frame):
        # RGB変換
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_frame)

        all_visited = False

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # 人差し指の先端の座標を取得
                h, w, _ = frame.shape
                finger_tip = (
                    int(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w),
                    int(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)
                )

                # ガイドラインの更新と描画
                all_visited = self._update_and_draw_segments(frame, finger_tip)

                # 手のランドマークを描画
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        else:
            # 手が検出されない場合は現在の状態を維持して描画
            self._draw_current_state(frame)

        # 完了メッセージの表示
        if all_visited:
            cv2.putText(
                frame,
                "Completed!",
                (int(w/2-100), 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                self.COLORS['completed'],
                2
            )

        return frame, all_visited

    def _update_and_draw_segments(self, frame, finger_tip):
        all_visited = True

        for i, (start, end) in enumerate(self.segments):
            if i < self.current_segment_index:
                color = self.COLORS['visited']
            elif i == self.current_segment_index:
                if self._is_near_line(finger_tip, start, end):
                    self.visited_segments[i] = True
                    self.current_segment_index += 1
                color = self.COLORS['current']
            else:
                color = self.COLORS['unvisited']
                all_visited = False

            cv2.line(frame, start, end, color, 5)

        if all(self.visited_segments):
            for start, end in self.segments:
                cv2.line(frame, start, end, self.COLORS['completed'], 5)
            return True

        return False

    def _draw_current_state(self, frame):
        for i, (start, end) in enumerate(self.segments):
            if i < self.current_segment_index:
                color = self.COLORS['visited']
            elif i == self.current_segment_index:
                color = self.COLORS['current']
            else:
                color = self.COLORS['unvisited']
            cv2.line(frame, start, end, color, 5)

def main():
    # カメラのセットアップ
    cap = cv2.VideoCapture(0)

    # HandGuidanceSystemのインスタンス化
    guidance_system = HandGuidanceSystem(
        num_segments=20,
        line_start=(100, 100),
        line_end=(400, 400)
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # フレームの処理
        frame, completed = guidance_system.process_frame(frame)

        # 結果の表示
        cv2.imshow("MR Hand Guidance", frame)

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # リソースの解放
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()