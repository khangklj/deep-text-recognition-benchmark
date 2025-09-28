import fire
import os
import lmdb
import cv2

import numpy as np


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(inputPath, gtFile, outputPath, checkValid=True):
    os.makedirs(outputPath, exist_ok=True)

    # start with a small map size (e.g., 256MB)
    env = lmdb.open(outputPath, map_size=256 * 1024 * 1024)

    cache = {}
    cnt = 1

    with open(gtFile, "r", encoding="utf-8") as data:
        datalist = data.readlines()

    nSamples = len(datalist)
    print(nSamples)

    for i in range(nSamples):
        imagePath, label = datalist[i].strip("\n").split(" ", 1)
        imagePath = os.path.join(inputPath, imagePath)

        if not os.path.exists(imagePath):
            print("%s does not exist" % imagePath)
            continue

        with open(imagePath, "rb") as f:
            imageBin = f.read()

        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print("%s is not a valid image" % imagePath)
                    continue
            except:
                print("error occured", i)
                with open(outputPath + "/error_image_log.txt", "a") as log:
                    log.write("%s-th image data occured error\n" % str(i))
                continue

        imageKey = "image-%09d".encode() % cnt
        labelKey = "label-%09d".encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()

        if cnt % 1000 == 0:
            try:
                writeCache(env, cache)
            except lmdb.MapFullError:
                # if out of space, double the map size
                curr_limit = env.info()["map_size"]
                new_limit = curr_limit * 2
                env.set_mapsize(new_limit)
                print(f"Resized map_size to {new_limit/1024/1024:.2f} MB")
                writeCache(env, cache)

            cache = {}
            print("Written %d / %d" % (cnt, nSamples))

        cnt += 1

    nSamples = cnt - 1
    cache["num-samples".encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print("Created dataset with %d samples" % nSamples)


if __name__ == "__main__":
    fire.Fire(createDataset)
