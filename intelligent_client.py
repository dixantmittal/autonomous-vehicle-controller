from __future__ import print_function

import argparse

import numpy as np

import network_wrapper
from carla.client import make_carla_client
from carla.sensor import Camera
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError


def run(host, port, online_training):
    n_episodes = args.episodes
    frames_per_episode = args.frames
    image_dims = (256, 128)
    network_wrapper.init(args.network)

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

        for episode in range(n_episodes):
            print('Starting new episode:', episode)
            client.start_episode(np.random.randint(number_of_player_starts - 1))

            # Iterate every frame in the episode.
            frame = 0
            count_diff = 0
            for frame in range(frames_per_episode):
                # Read the data produced by the server in this frame.
                measurements, sensor_data = client.read_data()

                # get the image and convert to numpy array
                image = sensor_data['CameraRGB']
                image = np.expand_dims(np.array(image.data), axis=0)
                image, _ = network_wrapper.normalize_data(X=image)

                if online_training:

                    # train and inference simultaneously
                    predicted_controls = network_wrapper.simultaneous_inference_and_learning(args.network, image,
                                                                                             measurements.player_measurements.autopilot_control.steer)

                    # because the autopilot follows a planned path,
                    # disconnect the episode if there's a disagreement between network and autopilot
                    if (predicted_controls[0].data[
                            0] - measurements.player_measurements.autopilot_control.steer) ** 2 > 0.5:
                        if count_diff < 10:
                            count_diff += 1
                        else:
                            break

                else:
                    predicted_controls = network_wrapper.inference(args.network, image)

                steer = predicted_controls[0].data[0]
                throttle = measurements.player_measurements.autopilot_control.throttle
                if measurements.player_measurements.forward_speed > 20:
                    throttle = min(0.5, throttle)

                print('Squared error: %.7f' % (steer - measurements.player_measurements.autopilot_control.steer) ** 2)

                # send control to simulator
                client.send_control(
                    steer=steer,
                    throttle=throttle,
                    brake=0.0,
                    hand_brake=False,
                    reverse=False)
            if frame == frames_per_episode - 1:
                network_wrapper.save(args.network)


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
        default=network_wrapper.control_net,
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
    argparser.add_argument('--online-training', dest='online_training', action='store_true')
    argparser.set_defaults(online_training=False)

    args = argparser.parse_args()

    while True:
        try:
            print('Data Aggregation: ', args.online_training)
            run(host=args.host, port=args.port, online_training=args.online_training)
            break
        except TCPConnectionError as error:
            print(error)
        except KeyboardInterrupt:
            print('\nCancelled by user. Bye!')
            break
