import threading

import hamming_codec
import numpy as np
import time

from abstract_modem import Encoder, StackedModem
import logging

logging.basicConfig(level=logging.INFO)


class NaiveModemSender(StackedModem):
    MIN_HZ = 2000
    MUL = (2.0 ** (1 / 6))
    CLK_NUM_LOW = 12
    CLK_NUM_HIGH = CLK_NUM_LOW + 1
    SYNC_DIV = 10

    def __init__(self, encoders, buf_size=1024, chunk_size=4410):
        super().__init__(encoders, buf_size, chunk_size)
        self.sin_range = np.linspace(0, 2 * np.pi, self.chunk_size)
        self.min_cycles = self.MIN_HZ // (44100 / chunk_size)
        self.sync_chunk = chunk_size // self.SYNC_DIV
        logging.info(f"sync_chunk {self.sync_chunk}")
        logging.info(
            f"waves base-hz {[self.cycles_for_num(i) * (44100 // chunk_size) for i in range(14)]}")

    def wave_by_hz(self, cycles: int):
        return np.sin(self.sin_range * cycles)

    def cycles_for_num(self, num: int):
        return int(self.min_cycles * (self.MUL ** num))

    def wave_by_num(self, num: int):
        base_cycles = self.cycles_for_num(num)
        tau = 0.7
        num = 1
        return np.sum([self.wave_by_hz(base_cycles)] +
                      [self.wave_by_hz(base_cycles * (3 + i)) * (tau ** i) for i
                       in
                       range(1, num)], axis=0)

    def data_to_signal(self, data: [int]) -> np.ndarray:
        # Use float32 because default is float64 and you will get garbage :P
        result = np.zeros(self.chunk_size * len(data), dtype=np.float32)
        for i, el in enumerate(data):
            wave = np.sum([self.wave_by_num(b) for b in range(12) if
                           (el & (1 << b)) != 0] + [
                              self.wave_by_num(self.CLK_NUM_LOW + (i % 2))],
                          axis=0)
            result[
            i * self.chunk_size: (i + 1) * self.chunk_size] = wave / np.max(
                np.abs(wave))
        return result

    def in_frame_decoder(self, frame: np.ndarray, prev_frame: np.ndarray):
        two_frame = np.append(prev_frame, frame)
        clk_diff = []
        subframes = two_frame.shape[0] // self.sync_chunk
        summary = []
        for offset in range(0, two_frame.shape[0] - self.sync_chunk + 1,
                            self.sync_chunk):
            small_fft = np.abs(
                np.fft.rfft(two_frame[offset:offset + self.sync_chunk]))
            low_power = small_fft[
                self.cycles_for_num(self.CLK_NUM_LOW) // self.SYNC_DIV]
            high_power = small_fft[
                self.cycles_for_num(self.CLK_NUM_HIGH) // self.SYNC_DIV]
            clk_diff.append(low_power - high_power)
            summary.append(1 if low_power - high_power > 0 else -1)
        finder_row = [1] * (subframes // 2) + [-1] * (subframes // 2)

        signal_check = np.array(
            [np.roll(finder_row, off) for off in range(subframes)],
            dtype=np.float32).transpose()

        finder_row = [1] * (subframes // 2) + [-1] * (subframes // 2)
        finder = np.array(
            [np.roll(finder_row, off) for off in range(subframes // 2)],
            dtype=np.float32).transpose()
        offset_qual = np.abs(np.matmul(clk_diff, finder))
        signal_lvl = np.max(offset_qual)
        signal_check_val = np.max(np.matmul(summary, signal_check))
        logging.debug(f"SIG LEVEL {signal_lvl} {signal_check_val}")
        if signal_lvl < 2 or signal_check_val < self.SYNC_DIV * 1.7:
            return [], self.NO_PACKET

        offset = np.argmax(offset_qual)
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
        return [val], self.IN_PACKET


class HammingByte(Encoder):
    def __init__(self):
        self.decodeMap = {}
        for i in range(1 << 12):
            decoded = int(hamming_codec.decode(i, 12), 2)
            self.decodeMap[i] = decoded, (
                    i != int(hamming_codec.encode(decoded, 8), 2))

    def encode(self, data):
        return [int(hamming_codec.encode(el, 8), 2) for el in data]

    def decode(self, data):
        decoded = []
        errors = 0
        for el in data:
            dec, error = self.decodeMap[el]
            decoded.append(dec)
            if error:
                errors += 1

            logging.debug(
                f"""{dec} from {el}={"".join([str(el // (1 << i) % 2) for i in range(12)])[
                                     ::-1]}, hasError={error}""")

        return decoded, errors


class PacketBorder:
    SIZE = 3

    def encode(self, data):
        return [i for i in range(self.SIZE, 0, -1)] + data + [i for i in
                                                              range(1,
                                                                    self.SIZE + 1)]

    def decode(self, data):
        if data[0] > self.SIZE or data[0] == 0:
            logging.warning(
                f"INVALID PACKET. PREAMBULE NOT FOUND {len(data)} {data[:5]}")
            return [], 1
        if data[-1] > self.SIZE or data[-1] == 0:
            logging.warning(
                f"INVALID PACKET. POSTAMBULE NOT FOUND {len(data)} {data[-5:]}")
            return [], 1
        return data[data[0]:-data[-1]], 0


if __name__ == '__main__':
    modem = NaiveModemSender([PacketBorder(), HammingByte()], 100, 441)
    modem_t = threading.Thread(target=modem.start)
    modem_t.start()

    payload = [i for i in range(1 << 8)]


    def _writer():
        def handler(received):
            if len(payload) != len(received):
                logging.warning(
                    f"different length! {len(payload)} {len(received)} {payload} {received}")
            differences = []
            for i, el_payload, el_received in zip(range(len(payload)), payload,
                                                  received):
                if el_payload != el_received:
                    differences.append((i, el_payload, el_received))
            print(f"Differences {differences}")

        modem.register_handler(handler)
        for i in range(5):
            modem.write_bytes(payload)


    time.sleep(1)
    writer = threading.Thread(target=_writer)
    writer.start()

    try:
        writer.join()
        modem_t.join()
    except KeyboardInterrupt:
        pass
    finally:
        print(modem.out)
