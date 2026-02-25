
try:
    from mmengine.logging import HistoryBuffer
    hb = HistoryBuffer()
    hb.update(10)
    hb.update(20)
    print(f"Attributes: {dir(hb)}")
    try:
        print(f"Mean: {hb.mean}")
    except:
        pass
except ImportError:
    print("HistoryBuffer not found in mmengine.logging")
except Exception as e:
    print(f"Error: {e}")
