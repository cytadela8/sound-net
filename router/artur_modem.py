import threading

import hamming_codec
import numpy as np
import time

from abstract_modem import AbstractModemSender


class NaiveModemSender(AbstractModemSender):
    MIN_HZ = 2000
    MUL = (2.0 ** (1 / 6))
    CLK_NUM_LOW = 12
    CLK_NUM_HIGH = CLK_NUM_LOW + 1
    SYNC_DIV = 10

    def __init__(self, buf_size=1024, chunk_size=4410):
        super().__init__(buf_size, chunk_size)
        self.sin_range = np.linspace(0, 2 * np.pi, self.chunk_size)
        self.min_cycles = self.MIN_HZ // (44100 / chunk_size)
        self.sync_chunk = chunk_size // self.SYNC_DIV
        print("sync_chunk", self.sync_chunk)
        self._player.register_handler(self.in_frame_handler)
        print("waves base-hz",
              [self.cycles_for_num(i) * (44100 // chunk_size) for i in
               range(14)])

    def wave_by_hz(self, cycles: int):
        return np.sin(self.sin_range * cycles)

    def cycles_for_num(self, num: int):
        return int(self.min_cycles * (self.MUL ** num))

    def wave_by_num(self, num: int):
        base_cycles = self.cycles_for_num(num)
        tau = 0.7
        num = 1
        return np.sum([self.wave_by_hz(base_cycles)]+
            [self.wave_by_hz(base_cycles * (3+i)) * (tau ** i) for i in
             range(1,num)], axis=0)

    def bytes_to_signal(self, data: bytes) -> np.ndarray:
        # Use float32 because default is float64 and you will get garbage :P
        result = np.zeros(self.chunk_size * len(data), dtype=np.float32)
        for i, el in enumerate(data):
            enc = int(hamming_codec.encode(el, 8), 2)
            wave = np.sum([self.wave_by_num(b) for b in range(12) if
                           (enc & (1 << b)) != 0] + [
                              self.wave_by_num(self.CLK_NUM_LOW + (i % 2))],
                          axis=0)
            result[
            i * self.chunk_size: (i + 1) * self.chunk_size] = wave / np.max(
                np.abs(wave))
        return result

    def in_frame_handler(self, frame: np.ndarray, prev_frame: np.ndarray):
        if prev_frame is None:
            return
        two_frame = np.append(prev_frame, frame)
        clk_diff = []
        subframes = two_frame.shape[0] // self.sync_chunk
        summary = ""
        for offset in range(0, two_frame.shape[0], self.sync_chunk):
            small_fft = np.abs(
                np.fft.rfft(two_frame[offset:offset + self.sync_chunk]))
            low_power = small_fft[
                self.cycles_for_num(self.CLK_NUM_LOW) // self.SYNC_DIV]
            high_power = small_fft[
                self.cycles_for_num(self.CLK_NUM_HIGH) // self.SYNC_DIV]
            clk_diff.append(low_power - high_power)
            summary += "1" if low_power - high_power > 0 else "0"
        finder_row = [1] * (subframes // 2) + [-1] * (subframes // 2)
        finder = np.array(
            [np.roll(finder_row, off) for off in range(subframes // 2)],
            dtype=np.float32).transpose()
        offset_qual = np.abs(np.matmul(clk_diff, finder))
        signal_lvl = np.max(offset_qual)
        print("SIG LEVEL", signal_lvl)

        offset = np.argmax(offset_qual)
        print(summary, offset)
        frame = two_frame[
                self.sync_chunk * offset:self.sync_chunk * offset + self.chunk_size]
        ft = np.abs(np.fft.rfft(frame))

        def has_wave(cycles: int):
            med = np.average([ft[self.cycles_for_num(self.CLK_NUM_LOW)],
                              ft[self.cycles_for_num(self.CLK_NUM_HIGH)]])
            return ft[cycles] > med

        present_waves = [has_wave(self.cycles_for_num(i)) for i in range(12)]
        val = 0
        for i in range(12):
            if present_waves[i]:
                val |= (1 << i)
        print(int(hamming_codec.decode(val, 12), 2), "from", val, "=",
              "".join(map((lambda x: "1" if x else "0"), present_waves))[::-1])


if __name__ == '__main__':
    modem = NaiveModemSender(10)
    modem_t = threading.Thread(target=modem.start)
    modem_t.start()


    def _writer():
        for i in range(256):
            modem.write_bytes(bytes([i for _ in range(1)]))


    time.sleep(2)
    writer = threading.Thread(target=_writer)
    writer.start()

    writer.join()
    modem_t.join()
