import torch.nn as nn
import torch


# sample code breaking down each step applied by the model
def explain():
    # batch size remains at 2 for the entire flow
    # input has 17 features and has sequence length of 10
    input = torch.randn(2, 17, 10)
    print(f"IN size:{input.shape}")  # torch.Size([2, 17, 10])
    m1 = nn.Conv1d(in_channels=17, out_channels=64, kernel_size=5, padding=0)
    # out1 now has 64 features and sequence length is 6.
    # Each of the 6 output positions is produced by a kernel sliding over 5 consecutive input positions.
    # Position 0 sees inputs 0-4, position 1 sees inputs 1-5, and so on (overlapping local windows)
    out1 = m1(input)
    print(f"Out size:{out1.shape}")  # torch.Size([2, 64, 6])
    m2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
    # out2 now has 128 features and sequence length is 6. The features from 6 entries is
    # now codified in sequence of 6, but each entry in the sequence now has 128 features
    out2 = m2(out1)
    print(f"Out size:{out2.shape}")  # torch.Size([2, 128, 6])
    m3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2)
    # out3 now has 256 features and sequence length is 6. The features from 6 entries is
    # now codified in sequence of 6, but each entry in the sequence now has 256 features
    out3 = m3(out2)
    print(f"Out size:{out3.shape}")  # torch.Size([2, 256, 6])
    an = nn.AdaptiveMaxPool1d(1)
    # for each of the 256 channels, the maximum activation across all 6 sequence positions
    # is selected independently. Result collapses sequence length from 6 to 1.
    # - before applying AdaptvieMaxPool1d output has 256 features and 6 entries in the sequence
    #   (hence shape: [2,256,6]
    # - for each feature maxpooling picks the maximum among all 6 entries. Hence this operation is done 256 times
    #   and the resulting output has the shape: [2,256,1]
    # (batch=2, channels=256, length=6)
    #          ↓
    # For channel_0: pick max across [pos_0, pos_1, pos_2, pos_3, pos_4, pos_5] → 1 value
    # For channel_1: pick max across [pos_0, pos_1, pos_2, pos_3, pos_4, pos_5] → 1 value
    # ...
    # For channel_255: pick max across [pos_0, pos_1, pos_2, pos_3, pos_4, pos_5] → 1 value
    #          ↓
    # (batch=2, channels=256, length=1)

    # One way to look at:
    # sequence:
    # [
    #    entry0:[feature0,feature1, feature2,......,feature255],
    #    entry1:[feature0,feature1, feature2,......,feature255],
    #    ...
    #    entry5:[feature0,feature1, feature2,......,feature255]
    # ]
    # maxpooling is picking max of feature0 , feature1, .. feature255 across all entries
    # PyTorch stores as (batch, 256, 6) — features before entries —
    out4 = an(out3)
    print(f"Out size:{out4.shape}")  # torch.Size([2, 256, 1])
    ff = nn.Flatten()
    # Flatten collapses all non-batch dimensions into one
    # (2, 256, 1) → (2, 256): 256*1=256, no change in values, only shape
    # if shape was (2,256,2) flatten would have resulted in output of shape:
    # (2,512)
    out5 = ff(out4)
    print(f"Out size:{out5.shape}")  # torch.Size([2, 256])
    # print(f"{out4}")


if __name__ == "__main__":
    explain()
