import threading
import winsound


class SoundManager(object):
    """
    Makes an annoying sound according to the input.
    """

    def __init__(self, sleep_detected=True):
        """
        :param sleep_detected: An indicator if the driver is sleeping to drowsing.
        """
        self.sleep_detected = sleep_detected

        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True  # Daemonize thread
        thread.start()  # Start the execution

    def run(self):
        """
        Makes an annoying sound to wake up a sleeping driver.
        """
        if self.sleep_detected:
            # play a constant annoying sound to wake the driver up.
            frequency = 2500  # Set Frequency To 2500 Hertz
            no_freq = 37
            duration = 250  # Set Duration To 250 ms == 1/4 second

            for j in range(8):
                if j % 2 == 0:
                    winsound.Beep(frequency, duration)
                else:
                    winsound.Beep(no_freq, duration)
        else:
            # play a number of beeps to alert the driver.
            frequency = 2500  # Set Frequency To 2500 Hertz
            duration = 2000  # Set Duration To 2000 ms == 2 second
            winsound.Beep(frequency, duration)
