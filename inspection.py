from typing import List, Optional, Callable, Dict, Tuple, Any
import time
from pathlib import Path
from datetime import datetime, timedelta
from hexss import check_packages
from hexss.image import Image
from hexss.constants import *

check_packages(
   'numpy', 'opencv-python', 'Flask', 'requests', 'pygame', 'pygame-gui',
   'tensorflow', 'keras', 'pyzbar', 'AutoInspection', 'matplotlib',
   'flatbuffers==23.5.26',
   auto_install=True
)

from hexss import json_load, close_port, system, username
from hexss.protocol.raspberrypi import IOController
from AutoInspection import AutoInspection, training
from AutoInspection.server import run_server

            
def io_func(data):
    
    def reset():
        if not io.get('Area').value and not io.get('EM').value:
            io.get('Switch Lamp').on()
            io.get('Cylinder 1-').on()
            io.get('Cylinder 2-').on()
        data['step'] = '-'

    def on_lamp():
        """if io.get('Proximity 1').value and io.get('Proximity 2').value and not io.get('Area').value and not io.get('EM').value:"""
        if io.get('Proximity 2').value and not io.get('Area').value and not io.get('EM').value:
            io.get('Switch Lamp').on()
            return True
        return False
    def cylinder_1_pull():
        if not io.get('Area').value and not io.get('EM').value:
            io.get('Cylinder 1-').on()
            return True
        return False
    def cylinder_1_push():
        if not io.get('Area').value and not io.get('EM').value:
            io.get('Cylinder 1+').on()
            return True
        return False
    
    
    def on_change(device, value):
        # print(f"[LOG] {device.name} -> {value}")
        
        if device.name == 'Cylinder 1+' and value == 1:
            io.get('Cylinder 1-').off()
        if device.name == 'Cylinder 1-' and value == 1:
            io.get('Cylinder 1+').off()
        if device.name == 'Cylinder 2+' and value == 1:
            io.get('Cylinder 2-').off()
        if device.name == 'Cylinder 2-' and value == 1:
            io.get('Cylinder 2+').off()


        if device.name in ['Proximity 1', 'Proximity 2']:
            if value == 0:
                io.get('Switch Lamp').off()
                if io.get('Proximity 1').value == 0 and io.get('Proximity 2').value == 0: # เอา part ออก
                    if data['step'] == 'WAIT_REMOVE_PART':
                        data['step'] = '-'
            if value == 1:    
                on_lamp()
            

        if device.name == 'Area':
            if value == 1:
                io.get('Switch Lamp').off()
                io.get('Cylinder 1+').off()
                io.get('Cylinder 1-').off()
            if value == 0:
                if data['step'] != 'WAIT_REMOVE_PART':
                    on_lamp()
            
        if device.name == 'EM':
            if value == 1:
                io.get('Switch Lamp').off()
                io.get('Cylinder 1-').off()
                io.get('Cylinder 2-').off()
                io.get('Cylinder 1+').off()
                io.get('Cylinder 2+').off()
            else:
                reset()
            

    def handle_simultaneous(events: List[Tuple[str, int]]):
        # print("simultaneous (0.2s):", events)
        if ('Switch L', 1) in events and ('Switch R', 1) in events:
            if io.get('EM').value or io.get('Area').value:
                io.get('Buzzer').blink(0.1, 0.1, n=2)
                return
            if io.get('Proximity 1').value == 0 and io.get('Proximity 2').value == 0:
                io.get('Cylinder 1-').on() #ถอยออก
                io.get('Buzzer').blink(0.1, 0.1, n=2)
                return
            
            if data['status'].get('set_qty',0) <= data['status'].get('pass_n',0):
                io.get('Buzzer').blink(0.1, 0.1, n=3)
                return
            """if io.get('Proximity 1').value and io.get('Proximity 2').value:"""
            if io.get('Proximity 2').value:
                if  data['step'] == 'PUSH': # หลังจากกำลังเข้าไป แล้ว error
                    cylinder_1_push()
                elif data['step'] == 'WAIT_REMOVE_PART': # หลังจากกำลังถอยออก แล้ว error
                    io.get('Switch Lamp').off()
                    cylinder_1_pull()
                    
                elif data['step'] != 'WAIT_REMOVE_PART':
                    io.get('Switch Lamp').blink(on_time=0.2, off_time=0.2)
                    io.get('Cylinder 1+').on()

                    fusion_states = [v['fusion_state'] for v in data['shared_state']['camera'].values()]                    
                    # IDLE, REQUESTED, PROCESSING, READY
                    if all(fusion_state == 'IDLE' for fusion_state in fusion_states):
                        io.get('Switch Lamp').blink(on_time=0.2, off_time=0.2)
                        data['step'] = 'PUSH'
                    
    
    io = IOController()

    io.input.add(5, "EM", bounce_time=0.02)
    io.input.add(12, "Switch L", pull_up=True, bounce_time=0.02)
    io.input.add(16, "Switch R", pull_up=True, bounce_time=0.02)
    io.input.add(20, "Area", bounce_time=0.02)
    io.input.add(6, "Proximity 1", pull_up=True, bounce_time=0.02)
    io.input.add(13, "Proximity 2", pull_up=True, bounce_time=0.02)
    io.input.add(19, "Cylinder 1 Reed Switch", pull_up=True, bounce_time=0.02)
    io.input.add(21, "Cylinder 2 Reed Switch", pull_up=True, bounce_time=0.02)

    io.output.add(4, 'Switch Lamp')
    io.output.add(18, 'Buzzer')
    io.output.add(22, 'Cylinder 1+')
    io.output.add(24, 'Cylinder 1-')
    io.output.add(17, 'Cylinder 2+')
    io.output.add(27, 'Cylinder 2-')
    io.output.add(23)
    io.output.add(25)
    
    io.on_change(on_change)
    io.simultaneous_events(handle_simultaneous, duration=0.2)
    io.start_server()
    
    reset()


    while data.get('play'):
        time.sleep(0.5)
        
        if data['step'] == 'PUSH':
            if io.get("Cylinder 1 Reed Switch").value == 1:
                data['step'] = '-'
                for k, v in data['shared_state']['camera'].items():
                    v['fusion_state'] = 'REQUESTED'
            
        # IDLE, REQUESTED, PROCESSING, READY
        fusion_states = [v['fusion_state'] for v in data['shared_state']['camera'].values()]
        if all(fusion_state == 'READY' for fusion_state in fusion_states):
            io.get('Switch Lamp').blink(on_time=0.1, off_time=0.1)
            for k, v in data['shared_state']['camera'].items():
                v['fusion_state'] = 'IDLE'
            
            im = Image.new("RGB", (2048, 1536), "#999")
            try: im.overlay(Image('http://127.0.0.1:3001/api/get_image?id=0&im=fused_result'), (0,0))
            except: ...
            try: im.overlay(Image('http://127.0.0.1:3001/api/get_image?id=4&im=fused_result'), (1024,0))
            except: ...
            try: im.overlay(Image('http://127.0.0.1:3001/api/get_image?id=6&im=fused_result'), (0,768))
            except: ...
            try: im.overlay(Image('http://127.0.0.1:3001/api/get_image?id=2&im=fused_result'), (1024,768))
            except: ...

            data['img_form_api'] = im.numpy()

            data['events'].append('change_image')
            data['events'].append('Predict')
            data['step'] = 'WAIT_PREDICT'
            continue
        
        if data['step'] == 'WAIT_PREDICT':
            if data['status']['res'] == 'Wait':
                continue
            
            elif data['status']['res'] == 'OK':
                data['step'] = 'WAIT_REMOVE_PART'
                io.get('Buzzer').blink(0.1, 0.1, n=1)
                io.get('Cylinder 2+').on()
                time.sleep(0.5)
                io.get('Cylinder 2-').on()
                time.sleep(0.1)
                if data['status'].get('set_qty',0) == data['status'].get('pass_n',0):
                    io.get('Buzzer').blink(0.1, 0.1, n=3)
                
                if io.get('Area').value == 0 and io.get('EM').value == 0:
                    io.get('Cylinder 1-').on()
                    io.get('Switch Lamp').off()

            
            elif data['status']['res'] == 'NG':
                data['step'] = '-'
                io.get('Buzzer').blink(on_time=0.2, off_time=0.3)
                data['events'].append('input_password')
                data['status']['enter_password_to_reset'] = False # True, False, 'wait'
                

        if data['status'].get('enter_password_to_reset') == True:
            data['status']['enter_password_to_reset'] = None   
            io.get('Buzzer').off()

            if not io.get('Area').value and not io.get('EM').value:
                io.get('Cylinder 1-').on()
                data['step'] = 'WAIT_REMOVE_PART'
                io.get('Switch Lamp').off()               
                


if __name__ == '__main__':
    from hexss.threading import Multithread


    config = json_load('config.json', {
        'projects_directory': r'C:\PythonProjects' if system == 'Windows' else f'/home/{username}/PythonProjects',
        'ipv4': '0.0.0.0',
        'port': 3000,
        'resolution_note': '1920x1080, 800x480',
        'resolution': '1920x1080' if system == 'Windows' else '800x480',
        'model_name': '-',
        'model_names': ["QC7-7990-000-Example", ],
        'fullscreen': True,
        'image_url': 'http://127.0.0.1:2002/image?source=video_capture&id=0',
    }, True)

    close_port(config['ipv4'], config['port'], verbose=False)

    # training
    # try:
    #     training(
    #         *config['model_names'],
    #         config={
    #             'projects_directory': config['projects_directory'],
    #             'batch_size': 32,
    #             'img_height': 180,
    #             'img_width': 180,
    #             'epochs': 5,
    #             'shift_values': [-4, -2, 0, 2, 4],
    #             'brightness_values': [-24, -12, 0, 12, 24],
    #             'contrast_values': [-12, -6, 0, 6, 12],
    #             'max_file': 20000,
    #         }
    #     )
    # except Exception as e:
    #     print(e)
    data = {
        'config': config,
        'model_name': config['model_name'],
        'model_names': config['model_names'],
        'events': [],
        'play': True,
        
        'shared_state':{
            'is_running': True,
            'ipv4': '0.0.0.0', 
            'port': 3001,
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
        },
        
        'img_form_api': None,
        'step': '-', # '-' ,'PUSH', '-', 'WAIT_PREDICT', '-' ,'WAIT_REMOVE_PART'
            
    }

    ui = AutoInspection(data)

    m = Multithread()
    # capture
    import capture_server
    import capture
    m.add_func(capture_server.run_server, args=(data['shared_state'],), join=False)
    for cam_id in data['shared_state']['camera']:
        m.add_func(capture.single_camera_worker, args=(data['shared_state'], cam_id), join=False)
    
    m.add_func(io_func, args=(data,))
    m.add_func(run_server, args=(data,), join=False)

    m.start()
    ui.run()
    m.join()
