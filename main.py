import vizdoom
import tensorflow as tf
from deepqnetwork import DQN
from frame_processing import preprocess, HistoryFrames


def create_game():
    game = vizdoom.DoomGame()
    game.load_config("ViZDoom/scenarios/health_gathering.cfg")
    game.set_screen_resolution(vizdoom.RES_160X120)
    game.set_screen_format(vizdoom.ScreenFormat.RGB24)
    game.set_window_visible(False)
    game.set_render_hud(False)
    game.set_render_crosshair(False)
    game.set_render_weapon(False)
    game.set_render_decals(False)
    game.set_render_particles(False)

    return game


if __name__ == "__main__":
    game = create_game()

    turn_left = (1, 0, 0)
    turn_right = (0, 1, 0)
    move_forward = (0, 0, 1)

    actions = [turn_left, turn_right, move_forward]

    res = [64, 64]
    history_length = 4
    input_shape = (res[0], res[1], history_length,)

    dqn = DQN(input_shape, actions)
    hf = HistoryFrames(res, history_length)

    def get_state():
        frame = game.get_state().screen_buffer
        frame = preprocess(frame, res, top_crop=40)
        hf.next(frame)

        return hf.state()

    episodes = 500
    game.init()
    for i in range(episodes):
        game.new_episode()
        hf.reset()

        state_prime = get_state()

        while not game.is_episode_finished():
            action_prime = dqn.choose_action(state_prime)
            reward_prime = game.make_action(action_prime)
            terminal = game.is_episode_finished()

            if not terminal:
                action = action_prime
                reward = reward_prime
                state = state_prime

                state_prime = get_state()
            
            dqn.save_experience_replay(state, action, reward, state_prime, terminal)

        print(f"Episode #{i + 1} Total Reward: {game.get_total_reward()}")

    game.close()
