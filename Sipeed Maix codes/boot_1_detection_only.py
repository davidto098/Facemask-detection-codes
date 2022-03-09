# object detector boot.py
# generated by maixhub.com

import sensor, image, lcd, time
import KPU as kpu
import gc, sys

def lcd_show_except(e):
    import uio
    err_str = uio.StringIO()
    sys.print_exception(e, err_str)
    err_str = err_str.getvalue()
    img = image.Image(size=(224,224))
    img.draw_string(0, 10, err_str, scale=1, color=(0xff,0x00,0x00))
    lcd.display(img)

def main(anchors, labels = None, model_addr="/sd/m.kmodel", sensor_window=(224, 224), lcd_rotation=3, sensor_hmirror=False, sensor_vflip=False):
    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.QVGA)
    sensor.set_windowing(sensor_window)
    sensor.set_hmirror(sensor_hmirror)
    sensor.set_vflip(sensor_vflip)
    sensor.run(1)

    lcd.init(type=1)

    lcd.clear(lcd.WHITE)

    if not labels:
        with open('labels.txt','r') as f:
            exec(f.read())
    if not labels:
        print("no labels.txt")
        img = image.Image(size=(320, 240))
        img.draw_string(90, 110, "no labels.txt", color=(255, 0, 0), scale=2)
        lcd.display(img)
        return 1
    try:
        img = image.Image("startup.jpg")
        lcd.display(img)
    except Exception:
        img = image.Image(size=(320, 240))
        img.draw_string(90, 110, "loading model...", color=(255, 255, 255), scale=2)
        lcd.display(img)

    try:
        task = None
        task = kpu.load(model_addr)
        kpu.init_yolo2(task, 0.5, 0.3, 5, anchors) # threshold:[0,1], nms_value: [0, 1]
        frame = 0
        predic_ant = 1
        while(True):

            img = sensor.snapshot()
            t = time.ticks_ms()

            objects = kpu.run_yolo2(task, img)
            t = time.ticks_ms() - t

            if objects:
                for obj in objects:
                    predic = obj.classid()

                    if predic != predic_ant:
                        predic_ant = obj.classid()
                        continue
                    pos = obj.rect()
                    activacion = False
                    if obj.classid() == 0:
                        color=(255, 139, 0)

                    elif obj.classid() == 1:
                        color=(255, 0, 0)

                    else:
                        color=(0, 255, 0)

                    img.draw_rectangle(pos, color=color)
                    img.draw_string(pos[0], pos[1], "%s : %.2f" %(labels[obj.classid()], obj.value()), scale=2, color=color)



                    predic_ant = obj.classid()
                    print(frame)
                    print(labels[obj.classid()])

            frame = frame + 1
            img.draw_string(0, 200, "t:%dms" %(t), scale=2, color=color)

        lcd.rotation(lcd_rotation)
            lcd.display(img)
    except Exception as e:
        raise e
    finally:
        if not task is None:
            kpu.deinit(task)


if __name__ == "__main__":
    try:
        labels = ["error_mask", "no_mask", "with_mask"]
        anchors = [5.1875, 6.5, 4.0625, 5.34375, 3.875, 4.75, 2.1875, 2.9375, 3.21875, 3.78125]
        # main(anchors = anchors, labels=labels, model_addr=0x300000, lcd_rotation=0)
        main(anchors = anchors, labels=labels, model_addr="/sd/m.kmodel")
    except Exception as e:
        sys.print_exception(e)
        lcd_show_except(e)
    finally:
        gc.collect()
