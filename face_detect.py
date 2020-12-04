import os
import common
import cv2
import click
import typing
import numpy as np
import base64
from typing import List, Dict

from face import detect, detect_image, HAS_CUDA, _get_model, detect_image_by_nparray
from model.DBFace import DBFace
from simple_face_alignment import align_face


def _draw_box(img: np.array, face_infos: List[Dict], output_filename):
    for face_info in face_infos:
        common.drawbbox(img, face_info)

    # common.imwrite("detect_result/" + common.file_name_no_suffix(filename) + ".draw.jpg", image)
    if output_filename != "":
        common.imwrite(output_filename, img)


def _do_align_face(raw_image: np.array, face_infos: List[common.BBox], output_filename: str):
    for inx, face_info in enumerate(face_infos):
        if not face_info:
            continue

        info = face_info.json
        landmark = info["landmark"]
        bbox = info["bbox"]
        aligned_face_image = align_face(raw_image, landmark, bbox)
        if output_filename == "":
            out_face_file = f"/tmp/aligned_face_{inx}.jpg"
        else:
            no_suffix_name = common.file_name_no_suffix(output_filename)
            base_dir = os.path.dirname(output_filename)
            out_face_file = f"{base_dir}/{no_suffix_name}_aligned_face_{inx}.jpg"

        cv2.imwrite(out_face_file, aligned_face_image)
        with open(out_face_file, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read())
            face_encode_str = encoded_image.decode('utf-8')
            face_info.aligned_face = face_encode_str

    return face_infos


def do_detect_image(filename, output_filename, align_face: bool = False):
    if filename == output_filename:
        raise ValueError("input and output filename can not be same")

    model = _get_model()
    try:
        raw_image = common.imread(filename)
        if raw_image is None:
            raise ValueError(f"{filename} is not a image file")
        face_infos = detect_image_by_nparray(model, raw_image)
        if align_face:
            face_infos = _do_align_face(raw_image, face_infos, output_filename)
        if face_infos is not None:
            _draw_box(raw_image, face_infos, output_filename)
        else:
            face_infos = {}
    except ValueError:
        face_infos = {}
    except cv2.error:
        face_infos = {}

    return list([face_info.json for face_info in face_infos])


def image_demo():
    dbface = DBFace()
    dbface.eval()

    if HAS_CUDA:
        dbface.cuda()

    dbface.load("model/dbface.pth")
    #    do_detect_image(dbface, "datas/selfie.jpg")
    #    do_detect_image(dbface, "datas/12_Group_Group_12_Group_Group_12_728.jpg")
    do_detect_image(dbface, "datas/head.jpg")
    do_detect_image(dbface, "datas/feichi2.jpeg")
    do_detect_image(dbface, "datas/ty1.jpg")
    do_detect_image(dbface, "datas/gxt1.png")
    do_detect_image(dbface, "datas/lq1.jpg")


@click.command()
@click.argument('filename', type=click.Path(exists=True, dir_okay=False))
@click.option("-d", "--output_filename", type=click.STRING, default="", help="output labeled image file")
@click.option("-a", "--align_face", is_flag=True, default=False)
def detect(filename, output_filename, align_face):
    ret = do_detect_image(filename, output_filename, align_face)
    print(ret)
    return ret


@click.command()
@click.argument('image_list_file', type=click.File('r'))
@click.option("-d", "--output_dir", type=click.STRING, default="")
def detect_image_listfile(image_list_file, output_dir):
    image_list = image_list_file.readlines()
    return detect_image_list(image_list, output_dir)


def detect_image_list(image_list: typing.List, output_dir, align_face: bool = False):
    if output_dir != "" and not os.path.exists(output_dir):
        raise ValueError(f"output folder ({output_dir}) is not existed")

    ret = {}
    for image_file_name in image_list:
        print(f"detect for {image_file_name}")
        output_file = ""
        if output_dir != "":
            base_name = common.file_name_no_suffix(image_file_name)
            output_file = f"{output_dir}/{base_name}.draw.jpg"

        face_info = do_detect_image(image_file_name, output_file, align_face)

        ret[image_file_name] = face_info

    return ret


@click.command()
@click.argument('image_folder', type=click.Path(exists=True, dir_okay=True))
@click.option("-d", "--output_dir", type=click.STRING, default="", help='folder for output labeled image')
@click.option("-r", "--recursive", is_flag=True, default=False, help="recursive for the sub folder")
@click.option("-a", "--align_face", is_flag=True, default=False)
def detect_image_folder(image_folder, output_dir, recursive, align_face):
    if recursive:
        allfiles = []
        for root, dirs, files in os.walk(image_folder):
            files_in_one_folder = [f"{root}/{file_name}" for file_name in files]
            allfiles.extend(files_in_one_folder)
    else:
        files = os.listdir(image_folder)
        allfiles = [f"{image_folder}/{file_name}" for file_name in files if
                    not os.path.isdir(f"{image_folder}/{file_name}")]

    results = detect_image_list(allfiles, output_dir, align_face)
    print(results)
    return results


@click.group()
def cli():
    pass


if __name__ == "__main__":
    # image_demo()
    # camera_demo()

    cli.add_command(detect)
    cli.add_command(detect_image_listfile)
    cli.add_command(detect_image_folder)
    cli()
