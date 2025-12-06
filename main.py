import cv2
import numpy as np
import threading
import time
from typing import List, Optional
from server import run_server


class ExposureFusionEngine:
    def __init__(self, contrast_weight=1.0, saturation_weight=1.0, exposure_weight=1.0):
        self.wc, self.ws, self.we = contrast_weight, saturation_weight, exposure_weight

    def _compute_contrast(self, gray_img):
        return np.abs(cv2.Laplacian(gray_img, cv2.CV_64F))

    def _compute_saturation(self, img):
        return np.std(img, axis=2)

    def _compute_exposedness(self, img):
        sigma = 0.2
        gauss_curve = np.exp(-0.5 * np.power(img - 0.5, 2) / np.power(sigma, 2))
        return np.prod(gauss_curve, axis=2)

    def _generate_weight_maps(self, images):
        weights = []
        epsilon = 1e-12
        for img in images:
            img_norm = img / 255.0
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
            contrast = self._compute_contrast(gray)
            saturation = self._compute_saturation(img_norm)
            exposedness = self._compute_exposedness(img_norm)
            weights.append(np.power(contrast, self.wc) * np.power(saturation, self.ws) * np.power(exposedness, self.we) + epsilon)
        sum_weights = np.sum(weights, axis=0)
        return [w / sum_weights for w in weights]

    def _gaussian_pyramid(self, img, levels):
        pyr = [img]
        for _ in range(levels - 1): pyr.append(cv2.pyrDown(pyr[-1]))
        return pyr

    def _laplacian_pyramid(self, img, levels):
        gauss_pyr = self._gaussian_pyramid(img, levels)
        lap_pyr = []
        for i in range(levels - 1):
            h, w = gauss_pyr[i].shape[:2]
            upsampled = cv2.pyrUp(gauss_pyr[i+1], dstsize=(w, h))
            lap_pyr.append(gauss_pyr[i] - upsampled)
        lap_pyr.append(gauss_pyr[-1])
        return lap_pyr

    def _reconstruct(self, pyramid):
        img = pyramid[-1]
        for i in range(len(pyramid) - 2, -1, -1):
            h, w = pyramid[i].shape[:2]
            img = cv2.pyrUp(img, dstsize=(w, h)) + pyramid[i]
        return img

    def fuse(self, images: List[np.ndarray]) -> Optional[np.ndarray]:
        if not images: return None
        shape = images[0].shape
        weights = self._generate_weight_maps(images)
        min_dim = min(shape[:2])
        num_levels = int(np.log2(min_dim)) - 2
        pyr_fusion = [np.zeros_like(img, dtype=np.float64) for img in self._gaussian_pyramid(images[0], num_levels)]

        for i in range(len(images)):
            img_float = images[i].astype(np.float64) / 255.0
            pyr_img = self._laplacian_pyramid(img_float, num_levels)
            pyr_weight = self._gaussian_pyramid(weights[i], num_levels)
            for level in range(num_levels):
                w_expanded = cv2.cvtColor(pyr_weight[level].astype(np.float32), cv2.COLOR_GRAY2BGR)
                pyr_fusion[level] += w_expanded * pyr_img[level]
        
        return (np.clip(self._reconstruct(pyr_fusion), 0, 1) * 255).astype(np.uint8)


def single_camera_worker(shared_state: dict, cam_id: str):
    """
    ทำงานเบื้องหลัง: อ่านกล้อง, ปรับ Exposure, ทำ Fusion
    และอัปเดตผลลัพธ์ลง shared_state เพื่อให้ Server ดึงไปแสดง
    """
    print(f"[Camera {cam_id}] Starting...")
    cam_config = shared_state['camera'][cam_id]
    cap = cv2.VideoCapture(int(cam_id))
    
    # Config Init
    settings = cam_config.get('setting', {})
    if 'CAP_PROP_FRAME_WIDTH' in settings: cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings['CAP_PROP_FRAME_WIDTH'])
    if 'CAP_PROP_FRAME_HEIGHT' in settings: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings['CAP_PROP_FRAME_HEIGHT'])
    
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3) # Default Auto

    fusion_engine = ExposureFusionEngine()
    BRACKET_SETTINGS = [5000, 1000, 20]
    # for test
    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    # cap.set(cv2.CAP_PROP_EXPOSURE, 5000)

    while shared_state['is_running'] and cam_config['is_running']:
        
        # --- COMMAND: REQUEST FUSION ---
        # คำสั่งนี้จะถูกส่งมาจาก Server (เมื่อกดปุ่มบนเว็บ)
        if cam_config['fusion_state'] == 'REQUESTED':
            cam_config['fusion_state'] = 'PROCESSING'
            print(f"[Camera {cam_id}] Processing Fusion...")
            
            captured_frames = []
            
            # 1. Start Bracketing (Manual Exposure)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)            
            
            # Get initial brightness
            ret, _frame = cap.read()
            last_mean = np.mean(_frame) if ret else 0
            
            for ev in BRACKET_SETTINGS:
                cap.set(cv2.CAP_PROP_EXPOSURE, ev)
                
                # Smart Wait (รอแสงนิ่ง)
                start_t = time.time()
                has_changed = False
                stable_count = 0
                prev_b = 0
                
                while (time.time() - start_t) < 2.0:
                    ret, frame = cap.read()
                    if not ret: break
                    
                    # Update Preview for Web
                    cam_config['latest_frame_data'] = (ret, frame)
                    
                    curr_b = np.mean(frame)
                    
                    # Check Change (>15%)
                    if not has_changed:
                        safe_old = last_mean if last_mean > 0.001 else 0.001
                        if abs(curr_b - safe_old)/safe_old > 0.15:
                            has_changed = True
                    
                    # Check Stability
                    if has_changed:
                        if abs(curr_b - prev_b) < 1.0: stable_count += 1
                        else: stable_count = 0
                        if stable_count >= 3: break
                    
                    prev_b = curr_b

                if cam_config['latest_frame_data'][1] is not None:
                    captured_frames.append(cam_config['latest_frame_data'][1])
                    last_mean = np.mean(cam_config['latest_frame_data'][1])

            # 2. Compute Fusion
            if captured_frames:
                result = fusion_engine.fuse(captured_frames)
                cam_config['fused_result'] = result
            
            # 3. Restore Auto Exposure
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
            cap.read() # flush
            
            cam_config['fusion_state'] = 'READY'
            print(f"[Camera {cam_id}] Fusion Done.")

        # --- NORMAL LOOP ---
        else:
            ret, frame = cap.read()
            if ret:
                cam_config['latest_frame_data'] = (ret, frame)
            else:
                time.sleep(0.1)
    
    cap.release()
    print(f"[Camera {cam_id}] Stopped.")


if __name__ == "__main__":
    
    shared_state = {
        'is_running': True,
        'ipv4': '0.0.0.0', 
        'port': 5000,
        'camera': {
            '0': { 
                'setting': {'CAP_PROP_FRAME_WIDTH': 1024, 'CAP_PROP_FRAME_HEIGHT': 768},
                'is_running': True, 
                'latest_frame_data': (None, None), 
                'fused_result': None, 
                'fusion_state': 'IDLE'
            },
            '2': { 
                'setting': {'CAP_PROP_FRAME_WIDTH': 1024, 'CAP_PROP_FRAME_HEIGHT': 768},
                'is_running': True, 
                'latest_frame_data': (None, None), 
                'fused_result': None, 
                'fusion_state': 'IDLE'
            },            
            '4': { 
                'setting': {'CAP_PROP_FRAME_WIDTH': 1024, 'CAP_PROP_FRAME_HEIGHT': 768},
                'is_running': True, 
                'latest_frame_data': (None, None), 
                'fused_result': None, 
                'fusion_state': 'IDLE'
            },            
            '6': { 
                'setting': {'CAP_PROP_FRAME_WIDTH': 1024, 'CAP_PROP_FRAME_HEIGHT': 768},
                'is_running': True, 
                'latest_frame_data': (None, None), 
                'fused_result': None, 
                'fusion_state': 'IDLE'
            }
        }
    }

    threads = []
    
    # 1. Start Web Server
    print("Starting Web Server...")
    t_server = threading.Thread(target=run_server, args=(shared_state,))
    t_server.start()
    threads.append(t_server)

    # 2. Start Camera Workers
    print("Starting Cameras...")
    for cam_id in shared_state['camera']:
        t_cam = threading.Thread(target=single_camera_worker, args=(shared_state, cam_id))
        t_cam.start()
        threads.append(t_cam)
    
    print(f"System Running. Access Dashboard at http://<PI_IP>:{shared_state['port']}")
    print("Press CTRL+C to stop.")

    # 3. Main Loop (Keep alive)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        shared_state['is_running'] = False
        
        for t in threads:
            t.join()
        print("Shutdown Complete.")