import argparse

import numpy as np
from tqdm import tqdm

from carla.client import make_carla_client
from carla.sensor import Camera
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError


def run(host, port):
    number_of_episodes = args.episodes
    frames_per_episode = args.frames

    with make_carla_client(host, port) as client:
        print('CarlaClient connected')

        for batch_no in range(args.batches):
            controls_center = []
            controls_left = []
            controls_right = []
            settings = CarlaSettings()
            settings.set(
                SynchronousMode=False,
                SendNonPlayerAgentsInfo=True,
                NumberOfVehicles=100,
                NumberOfPedestrians=100,
                WeatherId=batch_no//18)
            settings.randomize_seeds()

            camera_center = Camera('camera_center')
            camera_center.set_image_size(256, 128)
            camera_center.set_position(175, 0, 140)
            camera_center.set_rotation(-12, 0, 0)
            settings.add_sensor(camera_center)

            camera_left = Camera('camera_left')
            camera_left.set_image_size(256, 128)
            camera_left.set_position(175, 0, 140)
            camera_left.set_rotation(-12, 0, -30)
            if args.augmented: settings.add_sensor(camera_left)

            camera_right = Camera('camera_right')
            camera_right.set_image_size(256, 128)
            camera_right.set_position(175, 0, 140)
            camera_right.set_rotation(-12, 0, 30)
            if args.augmented: settings.add_sensor(camera_right)

            scene = client.load_settings(settings)
            number_of_player_starts = len(scene.player_start_spots)

            for episode in range(number_of_episodes):

                print('Starting new episode...:', episode)
                client.start_episode(np.random.randint(0, number_of_player_starts - 1))

                # Iterate every frame in the episode.
                for frame in tqdm(range(frames_per_episode)):

                    # Read the data produced by the server this frame.
                    measurements, sensor_data = client.read_data()

                    # if valid data, save image
                    for camera_name, image in sensor_data.items():
                        image.save_to_disk(args.dir + '/batch_{:0>3d}/{:s}/image_{:0>5d}.jpg'
                                           .format(args.batch_start + batch_no, camera_name,
                                                   episode * frames_per_episode + frame))

                    control = measurements.player_measurements.autopilot_control
                    controls_center.append((control.steer,
                                            control.throttle,
                                            measurements.player_measurements.forward_speed,
                                            control.brake))

                    if args.augmented:
                        controls_left.append((control.steer + 35 / 70,
                                              control.throttle,
                                              measurements.player_measurements.forward_speed,
                                              control.brake))

                        controls_right.append((control.steer - 35 / 70,
                                               control.throttle,
                                               measurements.player_measurements.forward_speed,
                                               control.brake,))

                    client.send_control(
                        steer=control.steer + 0.05 * np.random.randn(),
                        throttle=control.throttle,
                        brake=0.0,
                        hand_brake=False,
                        reverse=False)

            np.save(args.dir + '/batch_{:0>3d}/camera_center/controls.npy'.format(args.batch_start + batch_no),
                    controls_center)
            if args.augmented:
                np.save(args.dir + '/batch_{:0>3d}/camera_left/controls.npy'.format(args.batch_start + batch_no),
                        controls_left)
                np.save(args.dir + '/batch_{:0>3d}/camera_right/controls.npy'.format(args.batch_start + batch_no),
                        controls_right)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--host',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '--port',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--dir',
        default='data/dataset_00000')
    argparser.add_argument(
        '--episodes',
        default=10,
        type=int)
    argparser.add_argument(
        '--frames',
        default=1000,
        type=int)
    argparser.add_argument(
        '--batches',
        default=10,
        type=int)
    argparser.add_argument(
        '--batch-start',
        dest='batch_start',
        default=0,
        type=int)
    argparser.add_argument('--augmented', dest='augmented', action='store_true')
    argparser.set_defaults(augmented=False)

    args = argparser.parse_args()

    while True:
        try:
            run(host=args.host,
                port=args.port)
            print('Done.')
            break
        except TCPConnectionError as error:
            print(error)
        except KeyboardInterrupt:
            print('\nCancelled by user. Bye!')
            break
