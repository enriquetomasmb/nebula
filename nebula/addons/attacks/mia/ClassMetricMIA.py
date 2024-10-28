import numpy as np
import torch
from nebula.addons.attacks.mia.ShadowModelMIA import ShadowModelBasedAttack


class ClassMetricBasedAttack(ShadowModelBasedAttack):
    """
        Subclass of ShadowModelBasedAttack for conducting Class Metric-Based Membership Inference Attacks.
        This kind of MIA depends on training a single shadow model to create the threshold,
        which is slightly different from the usual metric-based MIAs.
    """
    def __init__(self, model, global_dataset, in_eval, out_eval, indexing_map, max_epochs, shadow_train,
                 shadow_test, num_s, attack_model_type, method_name):
        super().__init__(model, global_dataset, in_eval, out_eval, indexing_map, max_epochs, shadow_train,
                         shadow_test, num_s, attack_model_type)

        self.num_classes = 10

        self.s_in_outputs, self.s_in_labels = self.shadow_train_res  # Unpack shadow_in_eval and shadow_out_eval
        self.s_out_outputs, self.s_out_labels = self.shadow_test_res

        self.t_in_outputs, self.t_in_labels = self.in_eval_pre  # Unpack in_eval_pre and out_eval_pre
        self.t_out_outputs, self.t_out_labels = self.out_eval_pre

        # Move tensors to CPU before converting to NumPy
        self.s_in_outputs = self.s_in_outputs.cpu().detach().numpy()
        self.s_in_labels = self.s_in_labels.cpu().detach().numpy()
        self.s_out_outputs = self.s_out_outputs.cpu().detach().numpy()
        self.s_out_labels = self.s_out_labels.cpu().detach().numpy()
        self.t_in_outputs = self.t_in_outputs.cpu().detach().numpy()
        self.t_in_labels = self.t_in_labels.cpu().detach().numpy()
        self.t_out_outputs = self.t_out_outputs.cpu().detach().numpy()
        self.t_out_labels = self.t_out_labels.cpu().detach().numpy()

        self.s_in_conf = np.array([self.s_in_outputs[i, self.s_in_labels[i]] for i in range(len(self.s_in_labels))])
        self.s_out_conf = np.array([self.s_out_outputs[i, self.s_out_labels[i]] for i in range(len(self.s_out_labels))])
        self.t_in_conf = np.array([self.t_in_outputs[i, self.t_in_labels[i]] for i in range(len(self.t_in_labels))])
        self.t_out_conf = np.array([self.t_out_outputs[i, self.t_out_labels[i]] for i in range(len(self.t_out_labels))])

        self.s_in_entr = self._entr_comp(self.s_in_outputs)
        self.s_out_entr = self._entr_comp(self.s_out_outputs)
        self.t_in_entr = self._entr_comp(self.t_in_outputs)
        self.t_out_entr = self._entr_comp(self.t_out_outputs)

        self.s_in_m_entr = self._m_entr_comp(self.s_in_outputs, self.s_in_labels)
        self.s_out_m_entr = self._m_entr_comp(self.s_out_outputs, self.s_out_labels)
        self.t_in_m_entr = self._m_entr_comp(self.t_in_outputs, self.t_in_labels)
        self.t_out_m_entr = self._m_entr_comp(self.t_out_outputs, self.t_out_labels)

        self.methods_name = [method_name]

    def _log_value(self, probs, small_value=1e-30):
        return -np.log(np.maximum(probs, small_value))

    def _entr_comp(self, probs):
        return np.sum(np.multiply(probs, self._log_value(probs)), axis=1)

    def _m_entr_comp(self, probs, true_labels):
        log_probs = self._log_value(probs)
        reverse_probs = 1 - probs
        log_reverse_probs = self._log_value(reverse_probs)
        modified_probs = np.copy(probs)
        modified_probs[range(true_labels.size), true_labels] = reverse_probs[range(true_labels.size), true_labels]
        modified_log_probs = np.copy(log_reverse_probs)
        modified_log_probs[range(true_labels.size), true_labels] = log_probs[range(true_labels.size), true_labels]
        return np.sum(np.multiply(modified_probs, modified_log_probs), axis=1)

    def _thre_setting(self, tr_values, te_values):
        value_list = np.concatenate((tr_values, te_values))
        thre, max_acc = 0, 0
        for value in value_list:
            tr_ratio = np.sum(tr_values >= value) / (len(tr_values) + 0.0)
            te_ratio = np.sum(te_values < value) / (len(te_values) + 0.0)
            acc = 0.5 * (tr_ratio + te_ratio)
            if acc > max_acc:
                thre, max_acc = value, acc
        return thre

    def _mem_inf_thre(self, s_tr_values, s_te_values, t_tr_values, t_te_values):
        # perform membership inference attack by thresholding feature values: the feature can be prediction confidence,
        # (negative) prediction entropy, and (negative) modified entropy

        true_positives, false_positives = 0, 0

        for num in range(self.num_classes):
            thre = self._thre_setting(s_tr_values[self.s_in_labels == num], s_te_values[self.s_out_labels == num])

            true_positives += np.sum(t_tr_values[self.t_in_labels == num] >= thre)
            false_positives += np.sum(t_te_values[self.t_out_labels == num] >= thre)

        precision, recall, f1 = self.evaluate_metrics(true_positives, false_positives)

        return precision, recall, f1

    def mem_inf_benchmarks(self):
        if "Prediction Class Confidence" in self.methods_name:
            return self._mem_inf_thre(self.s_in_conf, self.s_out_conf, self.t_in_conf, self.t_out_conf)

        if "Prediction Class Entropy" in self.methods_name:
            return self._mem_inf_thre(-self.s_in_entr, -self.s_out_entr, -self.t_in_entr, -self.t_out_entr)

        if "Prediction Modified Entropy" in self.methods_name:
            return self._mem_inf_thre(-self.s_in_m_entr, -self.s_out_m_entr, -self.t_in_m_entr, -self.t_out_m_entr)

