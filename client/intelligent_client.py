from __future__ import print_function

import argparse
import os

import numpy as np

import network_handler
from carla.client import make_carla_client
from carla.sensor import Camera
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def run(host, port, dagger):
    n_episodes = args.episodes
    frames_per_episode = args.frames
    image_dims = (480, 240)

    with make_carla_client(host, port) as client:
        print('CarlaClient connected')

        # Start a new episode.
        settings = CarlaSettings()
        settings.set(
            SynchronousMode=False,
            SendNonPlayerAgentsInfo=True,
            NumberOfVehicles=25,
            NumberOfPedestrians=25,
            WeatherId=np.random.choice(14))
        settings.randomize_seeds()

        # add 1st camera
        camera0 = Camera('CameraRGB')
        camera0.set_image_size(image_dims[0], image_dims[1])
        camera0.set_position(175, 0, 140)
        camera0.set_rotation(-12, 0, 0)
        settings.add_sensor(camera0)

        # Choose one player start at random.
        scene = client.load_settings(settings)
        number_of_player_starts = len(scene.player_start_spots)

        expert_controls = []
        for episode in range(n_episodes):
            print('Starting new episode:', episode)
            client.start_episode(np.random.randint(number_of_player_starts - 1))

            # Iterate every frame in the episode.
            for frame in range(frames_per_episode):
                # Read the data produced by the server in this frame.
                measurements, sensor_data = client.read_data()

                image = sensor_data['CameraRGB']

                if dagger:
                    image.save_to_disk(args.dir + '/image_{:0>5d}.jpg'.format(episode * frames_per_episode + frame))
                    control = measurements.player_measurements.autopilot_control
                    expert_controls.append((control.steer + 30 / 70,
                                            control.throttle,
                                            control.brake,
                                            measurements.player_measurements.forward_speed))

                image = np.expand_dims(np.array(image.data), axis=0)
                image, _ = network_handler.flip_data(X=image)
                image, _ = network_handler.normalize_data(X=image)
                predicted_controls = network_handler.predict_controls(argparser.parse_args().network, image)[0]
                _, predicted_controls = network_handler.unnormalize_data(y=predicted_controls)
                _, predicted_controls = network_handler.flip_data(y=predicted_controls)
                steer, throttle = predicted_controls[0], predicted_controls[1]

                print('steer: ', steer, 'throttle: ', throttle)

                if measurements.player_measurements.forward_speed > 20:
                    throttle = 0

                # send control to simulator
                client.send_control(
                    steer=steer,
                    throttle=throttle,
                    brake=0.0,
                    hand_brake=False,
                    reverse=False)

                if dagger:
                    np.save(args.dir + '/controls.npy', expert_controls)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--host',
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
        '--network',
        default=network_handler.light_imitation_net,
        help='Network to use')
    argparser.add_argument(
        '--dir',
        default='data/dataset_dagger')
    argparser.add_argument(
        '--episodes',
        default=10,
        type=int)
    argparser.add_argument(
        '--frames',
        default=1000,
        type=int)
    argparser.add_argument('--dagger', dest='dagger', action='store_true')
    argparser.set_defaults(dagger=False)

    args = argparser.parse_args()

    while True:
        try:
            print('Data Aggregation: ', args.dagger)
            run(host=args.host, port=args.port, dagger=args.dagger)
            break
        except TCPConnectionError as error:
            print(error)
        except KeyboardInterrupt:
            print('\nCancelled by user. Bye!')
            break
