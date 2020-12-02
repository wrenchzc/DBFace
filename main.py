import os
import common
import cv2
import click
import typing
from tempfile import TemporaryFile

from face import detect, detect_image, HAS_CUDA, _get_model
from model.DBFace import DBFace


def do_detect_image(filename, output_filename):
    if filename == output_filename:
        raise ValueError("input and output filename can not be same")

    model = _get_model()
    try:
        objs = detect_image(model, filename)
        if objs is not None:
            image = common.imread(filename)
            for obj in objs:
                common.drawbbox(image, obj)

            print(objs)
            # common.imwrite("detect_result/" + common.file_name_no_suffix(filename) + ".draw.jpg", image)
            if output_filename != "":
                common.imwrite(output_filename, image)
        else:
            objs = {}
    except ValueError:
        objs = {}
    except cv2.error:
        objs = {}

    return objs


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


def camera_demo():
    dbface = DBFace()
    dbface.eval()

    if HAS_CUDA:
        dbface.cuda()

    dbface.load("model/dbface.pth")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ok, frame = cap.read()

    while ok:
        objs = detect(dbface, frame)

        for obj in objs:
            common.drawbbox(frame, obj)

        cv2.imshow("demo DBFace", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        ok, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()


@click.command()
@click.argument('filename', type=click.Path(exists=True, dir_okay=False))
@click.option("-d", "--output_filename", type=click.STRING, default="", help="output labeled image file")
def detect(filename, output_filename):
    do_detect_image(filename, output_filename)


@click.command()
@click.argument('image_list_file', type=click.File('r'))
@click.option("-d", "--output_dir", type=click.STRING, default="")
def detect_image_listfile(image_list_file, output_dir):
    image_list = image_list_file.readlines()
    return detect_image_list(image_list, output_dir)


def detect_image_list(image_list: typing.List, output_dir):
    if output_dir != "" and not os.path.exists(output_dir):
        raise ValueError(f"output folder ({output_dir}) is not existed")

    ret = {}
    for image_file_name in image_list:
        print(f"detect for {image_file_name}")
        output_file = ""
        if output_dir != "":
            base_name = common.file_name_no_suffix(image_file_name)
            output_file = f"{output_dir}/{base_name}.draw.jpg"

        face_info = do_detect_image(image_file_name, output_file)

        ret[image_file_name] = face_info

    return ret


@click.command()
@click.argument('image_folder', type=click.Path(exists=True, dir_okay=True))
@click.option("-d", "--output_dir", type=click.STRING, default="", help='folder for output labeled image')
@click.option("-r", "--recursive", is_flag=True, default=False, help="recursive for the sub folder")
def detect_image_folder(image_folder, output_dir, recursive):
    if recursive:
        allfiles = []
        for root, dirs, files in os.walk(image_folder):
            files_in_one_folder = [f"{root}/{file_name}" for file_name in files]
            allfiles.extend(files_in_one_folder)
    else:
        files = os.listdir(image_folder)
        allfiles = [f"{image_folder}/{file_name}" for file_name in files if
                    not os.path.isdir(f"{image_folder}/{file_name}")]

    results = detect_image_list(allfiles, output_dir)
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
