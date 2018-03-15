from __future__ import print_function

import argparse

import numpy as np

import nn_helper
import utils
from carla.client import make_carla_client
from carla.sensor import Camera
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError

# Constants
# Environment Settings
settings = CarlaSettings()
settings.set(SynchronousMode=False,
             SendNonPlayerAgentsInfo=True,
             NumberOfVehicles=25,
             NumberOfPedestrians=25,
             WeatherId=np.random.choice(18))
settings.randomize_seeds()

# Camera Settings
camera0 = Camera('CameraRGB')
camera0.set_image_size(256, 128)
camera0.set_position(175, 0, 140)
camera0.set_rotation(-12, 0, 0)
settings.add_sensor(camera0)


def save(loss_history):
    if args.online_training:
        np.save('loss_history_online_training.npy', np.array(loss_history))
        utils.save(args.network)


def run():
    loss_history = []

    # create carla connection
    client = make_carla_client(args.host, args.port)
    print('Carla connected')

    # Load Settings
    scene = client.load_settings(settings)

    for episode in range(args.episodes):
        # init network from scratch
        nn_helper.init(args.architecture)

        # select a starting position for the car
        print('Starting new episode:', episode + 1)
        client.start_episode(episode % scene.player_start_spots)

        for frame in range(args.frames):
            # Read the data produced by the server in this frame.
            measurements, sensor_data = client.read_data()

            # get the image and convert to normalized numpy array
            image = np.array(sensor_data['CameraRGB'].data)
            image = utils.normalize_image(image)

            # get autopilot controls
            autopilot = measurements.player_measurements.autopilot_control.steer * 70

            controls = nn_helper.inference(args.network, image, args.online_training, autopilot)

            steer = controls.data[0]

            loss = abs(steer - autopilot)
            print('Loss: %.5f' % loss)
            loss_history.append(loss)

            # send control to simulator
            client.send_control(
                steer=steer / 70,
                throttle=0.5,
                brake=0.0,
                hand_brake=False,
                reverse=False)

            # save progress
            save(loss_history)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-h', '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--architecture',
        default=nn_helper.control_net,
        help='Network Architecture to use')
    argparser.add_argument(
        '-e', '--episodes',
        default=10,
        type=int)
    argparser.add_argument(
        '-f', '--frames',
        default=1000,
        type=int)
    argparser.add_argument(
        '-ot', '--online-training',
        dest='online_training',
        action='store_true')
    argparser.set_defaults(online_training=False)

    args = argparser.parse_args()

    try:
        print('ONLINE TRAINING: ', args.online_training)
        run()
    except TCPConnectionError as error:
        print(error)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
