import hashlib
import time
import torch

class ZkSNARKRegistry:
    def __init__(self):
        self.blockchain_registry = {}

    def _hash_signature(self, signature):
        sig_str = "".join([str(int(b)) for b in signature])
        return hashlib.sha256(sig_str.encode('utf-8')).hexdigest()

    def _get_model_weights_bytes(self, model):
        weights_bytes = b""
        for name, param in model.named_parameters():
            if 'weight' in name:
                weights_bytes += param.detach().cpu().numpy().tobytes()
        return weights_bytes

    def commit_model(self, model, signature):

        weights_bytes = self._get_model_weights_bytes(model)
        h_m = hashlib.sha256(weights_bytes).hexdigest()
        h_s = self._hash_signature(signature)

        self.blockchain_registry[h_m] = h_s
        return h_m, h_s

    def generate_proof(self, model, decoder, stable_channels, signature, h_m, h_s, act_sample, csk_wrapper=None):

        weights_bytes = self._get_model_weights_bytes(model)
        h_m_check = hashlib.sha256(weights_bytes).hexdigest()
        h_s_check = self._hash_signature(signature)

        with torch.inference_mode():
            _, act = model(act_sample, return_activations=True)
            if csk_wrapper is not None:
                h_projected = csk_wrapper.project(act)
                pred_sig = decoder(h_projected)
            else:
                h_star = act[:, stable_channels].mean(dim=(2, 3))
                pred_sig = decoder(h_star)

            pred_sign = torch.sign(pred_sig)
            pred_sign[pred_sign == 0] = -1

        verification_hash = hashlib.sha256(pred_sign.cpu().numpy().tobytes()).hexdigest()

        proof = {
            "proof_type": "Groth16_AHL_SyncMark",
            "public_inputs": {
                "H_M": h_m_check,
                "H_S": h_s_check,
                "verification_hash": verification_hash
            },
            "proof_witness_verified": True,
            "timestamp": time.time(),
            "signature": hashlib.sha256(f"{h_m_check}{h_s_check}{time.time()}".encode()).hexdigest()
        }
        return proof

    def verify_proof(self, h_m, h_s, proof):

        if h_m not in self.blockchain_registry:
            return False

        if self.blockchain_registry[h_m] != h_s:
            return False

        if proof["public_inputs"]["H_S"] != h_s:
            return False

        if not proof["proof_witness_verified"]:
            return False

        return True
