from harvesters.core import Harvester
import cv2
h = Harvester()

# Load CTI file to communicate
h.add_file('/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti')
h.update()
print(h.device_info_list)

# Prepare for image acquisition
ia = h.create()
# Start image acquisition
ia.start()
while True:
    with ia.fetch() as buffer:
        component = buffer.payload.components[0]
        _2d = component.data.reshape(component.height,component.width)
        img = _2d
        cv2.imshow('img',img)
        cv2.imwrite('test.tiff',img)
        cv2.waitKey(10)
        # The buffer will automatically be queued
ia.stop()
ia.destroy()
h.reset()