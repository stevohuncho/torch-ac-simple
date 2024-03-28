import torch
import torch_ac
import time
from typing import Literal
from .model import ACModel
from torch_ac.utils.penv import ParallelEnv
from torch_ac_simple.utils import device
import torch_ac_simple.utils as utils
from .config import Config
import tensorboardX



class Agent:
    """An self.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, env, model_name, argmax=False, num_envs=1, seed=1, use_memory=False, use_text=False):
        utils.seed(seed)
        self.model_dir = utils.get_model_dir(model_name)
        envs = []
        for i in range(num_envs):
            envs.append(utils.make_env(env, seed + 10000 * i))
        self.envs = envs
        self.env = ParallelEnv(envs)
        obs_space, self.preprocess_obss = format.get_obss_preprocessor(self.env.observation_space)
        self.acmodel = ACModel(obs_space, self.env.action_space, use_memory=use_memory, use_text=use_text)
        self.argmax = argmax
        self.num_envs = num_envs
        self.seed = seed
        self.txt_logger = utils.get_txt_logger(self.model_dir)
        self.csv_file, self.csv_logger = utils.get_csv_logger(self.model_dir)
        self.tb_writer = tensorboardX.SummaryWriter(self.model_dir)
        if self.acmodel.recurrent:
            self.memories = torch.zeros(self.num_envs, self.acmodel.memory_size, device=device)

    def get_actions(self, obss):
        preprocessed_obss = self.preprocess_obss(obss, device=device)

        with torch.no_grad():
            if self.acmodel.recurrent:
                dist, _, self.memories = self.acmodel(preprocessed_obss, self.memories)
            else:
                dist, _ = self.acmodel(preprocessed_obss)

        if self.argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()

        return actions.cpu().numpy()

    def get_action(self, obs):
        return self.get_actions([obs])[0]

    def analyze_feedbacks(self, rewards, dones):
        if self.acmodel.recurrent:
            masks = 1 - torch.tensor(dones, dtype=torch.float, device=device).unsqueeze(1)
            self.memories *= masks

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])
    
    def eval(self, eps: int):
        # Load environments
        print("Environments loaded\n")

        # Initialize logs
        logs = {"num_frames_per_episode": [], "return_per_episode": []}

        # Run agent
        start_time = time.time()

        obss = self.env.reset()

        log_done_counter = 0
        log_episode_return = torch.zeros(self.num_envs, device=device)
        log_episode_num_frames = torch.zeros(self.num_envs, device=device)

        while log_done_counter < eps:
            actions = self.get_actions(obss)
            obss, rewards, terminateds, truncateds, _ = self.env.step(actions)
            dones = tuple(a | b for a, b in zip(terminateds, truncateds))
            self.analyze_feedbacks(rewards, dones)

            log_episode_return += torch.tensor(rewards, device=device, dtype=torch.float)
            log_episode_num_frames += torch.ones(self.num_envs, device=device)

            for i, done in enumerate(dones):
                if done:
                    log_done_counter += 1
                    logs["return_per_episode"].append(log_episode_return[i].item())
                    logs["num_frames_per_episode"].append(log_episode_num_frames[i].item())

            mask = 1 - torch.tensor(dones, device=device, dtype=torch.float)
            log_episode_return *= mask
            log_episode_num_frames *= mask

        end_time = time.time()

        # Print logs

        num_frames = sum(logs["num_frames_per_episode"])
        fps = num_frames / (end_time - start_time)
        duration = int(end_time - start_time)
        return_per_episode = utils.synthesize(logs["return_per_episode"])
        num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

        print("F {} | FPS {:.0f} | D {} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {}"
                .format(num_frames, fps, duration,
                        *return_per_episode.values(),
                        *num_frames_per_episode.values()))
        
    def train(self, frames: int, algo_type: Literal['ppo', 'a2c'], algo_config: Config = Config(), log_interval: int = 1, save_interval: int = 1):
        txt_logger = utils.get_txt_logger(self.model_dir)
        csv_file, csv_logger = utils.get_csv_logger(self.model_dir)
        tb_writer = tensorboardX.SummaryWriter(self.model_dir)

        try:
            status = utils.get_status(self.model_dir)
        except OSError:
            status = {"num_frames": 0, "update": 0}

        if "vocab" in status:
            self.preprocess_obss.vocab.load_vocab(status["vocab"])
        if "model_state" in status:
            self.acmodel.load_state_dict(status["model_state"])
        self.acmodel.to(device)
        txt_logger.info("Model loaded\n")
        txt_logger.info("{}\n".format(self.acmodel))

        if algo_type == "a2c":
            algo = torch_ac.A2CAlgo(self.envs, self.acmodel, device, algo_config.frames_per_proc, algo_config.discount, algo_config.lr, algo_config.gae_lambda,
                                    algo_config.entropy_coef, algo_config.value_loss_coef, algo_config.max_grad_norm, algo_config.recurrence,
                                    algo_config.optim_alpha, algo_config.optim_eps, self.preprocess_obss, algo_config.reshape_reward)
        elif algo_type == "ppo":
            algo = torch_ac.PPOAlgo(self.envs, self.acmodel, device, algo_config.frames_per_proc, algo_config.discount, algo_config.lr, algo_config.gae_lambda,
                                    algo_config.entropy_coef, algo_config.value_loss_coef, algo_config.max_grad_norm, algo_config.recurrence,
                                    algo_config.optim_eps, algo_config.clip_eps, algo_config.epochs, algo_config.batch_size, self.preprocess_obss, algo_config.reshape_reward)
        else:
            raise ValueError("Incorrect algorithm name: {}".format(algo_type))

        if "optimizer_state" in status:
            algo.optimizer.load_state_dict(status["optimizer_state"])

        num_frames = status["num_frames"]
        update = status["update"]
        start_time = time.time()

        while num_frames < frames:
            # Update model parameters
            update_start_time = time.time()
            exps, logs1 = algo.collect_experiences()
            logs2 = algo.update_parameters(exps)
            logs = {**logs1, **logs2}
            update_end_time = time.time()

            num_frames += logs["num_frames"]
            update += 1

            # Print logs

            if update % log_interval == 0:
                fps = logs["num_frames"] / (update_end_time - update_start_time)
                duration = int(time.time() - start_time)
                return_per_episode = utils.synthesize(logs["return_per_episode"])
                rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
                num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

                header = ["update", "frames", "FPS", "duration"]
                data = [update, num_frames, fps, duration]
                header += ["return_" + key for key in return_per_episode.keys()]
                data += rreturn_per_episode.values()
                header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
                data += num_frames_per_episode.values()
                header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
                data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

                txt_logger.info(
                    "U {} | F {:06} | FPS {:04.0f} | D {} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
                    .format(*data))

                header += ["return_" + key for key in return_per_episode.keys()]
                data += return_per_episode.values()

                if status["num_frames"] == 0:
                    csv_logger.writerow(header)
                csv_logger.writerow(data)
                csv_file.flush()

                for field, value in zip(header, data):
                    tb_writer.add_scalar(field, value, num_frames)

            # Save status

            if save_interval > 0 and update % save_interval == 0:
                status = {"num_frames": num_frames, "update": update,
                            "model_state": self.acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
                if hasattr(self.preprocess_obss, "vocab"):
                    status["vocab"] = self.preprocess_obss.vocab.vocab
                utils.save_status(status, self.model_dir)
                txt_logger.info("Status saved")