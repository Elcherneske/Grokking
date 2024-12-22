import torch.nn as nn
import torch
import pickle


class DatasetFormer(nn.Module):
    def __init__(
            self,
            prime_number = 97,
            op_number = 13,
            d_model = 128
    ):
        super().__init__()
        self.d_model = d_model
        self.prime_number = prime_number
        self.op_number = op_number
        self.number_emb = nn.Embedding(num_embeddings=prime_number, embedding_dim=self.d_model)
        self.op_emb = nn.Embedding(num_embeddings=op_number, embedding_dim=self.d_model)            # '=' is 0
        self.data = []

    def formData(self):
        eq_op_emb = self.op_emb(torch.tensor(0))
        for x in torch.arange(0, self.prime_number, 1):
            for y in torch.arange(0, self.prime_number, 1):
                # op 1
                result = int(x + y) % self.prime_number
                self.data.append(
                    {'x': self.number_emb(x).detach().numpy(), 'op': self.op_emb(torch.tensor(1)).detach().numpy(),
                     'y': self.number_emb(y).detach().numpy(), '=': eq_op_emb.detach().numpy(),
                     'result': result})

                # op 2
                result = int(x - y) % self.prime_number
                self.data.append(
                    {'x': self.number_emb(x).detach().numpy(), 'op': self.op_emb(torch.tensor(2)).detach().numpy(),
                     'y': self.number_emb(y).detach().numpy(), '=': eq_op_emb.detach().numpy(),
                     'result': result})

                # op 3
                result = int(x + y) % self.prime_number
                self.data.append(
                    {'x': self.number_emb(x).detach().numpy(), 'op': self.op_emb(torch.tensor(3)).detach().numpy(),
                     'y': self.number_emb(y).detach().numpy(), '=': eq_op_emb.detach().numpy(),
                     'result': result})

                # op 4
                if (y != 0):
                    result = int(x/y) % self.prime_number
                self.data.append(
                    {'x': self.number_emb(x).detach().numpy(), 'op': self.op_emb(torch.tensor(4)).detach().numpy(),
                     'y': self.number_emb(y).detach().numpy(), '=': eq_op_emb.detach().numpy(),
                     'result': result})

                # op 5
                if y % 2 == 1:
                    result = int(x/y) % self.prime_number
                else:
                    result = int(x - y) % self.prime_number
                self.data.append(
                    {'x': self.number_emb(x).detach().numpy(), 'op': self.op_emb(torch.tensor(5)).detach().numpy(),
                     'y': self.number_emb(y).detach().numpy(), '=': eq_op_emb.detach().numpy(),
                     'result': result})

                # op 6
                result = int(x**2 + y**2) % self.prime_number
                self.data.append(
                    {'x': self.number_emb(x).detach().numpy(), 'op': self.op_emb(torch.tensor(6)).detach().numpy(),
                     'y': self.number_emb(y).detach().numpy(), '=': eq_op_emb.detach().numpy(),
                     'result': result})

                # op 7
                result = int(x**2 + y**2 + x * y) % self.prime_number
                self.data.append(
                    {'x': self.number_emb(x).detach().numpy(), 'op': self.op_emb(torch.tensor(7)).detach().numpy(),
                     'y': self.number_emb(y).detach().numpy(), '=': eq_op_emb.detach().numpy(),
                     'result': result})

                # op 8
                result = int(x**2 + y**2 + x * y + x) % self.prime_number
                self.data.append(
                    {'x': self.number_emb(x).detach().numpy(), 'op': self.op_emb(torch.tensor(8)).detach().numpy(),
                     'y': self.number_emb(y).detach().numpy(), '=': eq_op_emb.detach().numpy(),
                     'result': result})

                # op 9
                result = int(x**3 + x * y) % self.prime_number
                self.data.append(
                    {'x': self.number_emb(x).detach().numpy(), 'op': self.op_emb(torch.tensor(9)).detach().numpy(),
                     'y': self.number_emb(y).detach().numpy(), '=': eq_op_emb.detach().numpy(),
                     'result': result})

                # op 10
                result = int(x**3 + x * y * y + y) % self.prime_number
                self.data.append(
                    {'x': self.number_emb(x).detach().numpy(), 'op': self.op_emb(torch.tensor(10)).detach().numpy(),
                     'y': self.number_emb(y).detach().numpy(), '=': eq_op_emb.detach().numpy(),
                     'result': result})

    def dump_data(self, outputFilename):
        with open(outputFilename, 'wb') as file:
            pickle.dump(self.data, file)




if __name__ == '__main__':
    former = DatasetFormer(prime_number=97, op_number=13, d_model=128)
    former.formData()
    former.dump_data('data.pkl')
