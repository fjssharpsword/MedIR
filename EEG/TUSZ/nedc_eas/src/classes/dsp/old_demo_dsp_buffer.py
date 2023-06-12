import numpy as np

class DemoDSPBuffer:
    def __init__(self,
                 ratio_a,
                 use_zero_padding,
                 taper=None):

        self.ratio_t_scale_to_slice_dist = ratio_a
        self.use_zero_padding = use_zero_padding

        if taper is not None:
            self.use_taper = True
            self.taper = taper
        else:
            self.use_taper = False

    def set_data(self,
                  data_a):
        self.data = data_a
    
    def set_frame(self,
                  left_index_a,
                  right_index_a):
        self.frame = self.data[left_index_a:right_index_a]
        self.frame_size = len(self.frame)

    def set_performance_ratio(self,
                              ratio_a):
        self.ratio_t_scale_to_slice_dist = float(ratio_a)

    # method: setup_for_make_window_array
    #
    # arguments:
    #  - num_non_zero_elems_a: the length of non-zero part of a single window
    #  - zero_padding: (optional) the number of zeros to be concatenated
    #
    # returns: None
    #
    # sets up buffer to later make array of windows by:
    # 1) updating self.zero_padding
    # 1) updating self.index_pairs
    # 2) check for small_window_flag
    #
    def setup_for_make_window_array(self,
                                    time_scale_a,
                                    num_non_zero_elems_a,
                                    zero_padding_a=None):

        self.calc_size = num_non_zero_elems_a

        # if the application will need zero padding (i.e. fft, but not
        # rms), initialize array of zeroes to be used in zero-padding
        #
        if self.use_zero_padding:
            self.zero_padding = np.zeros(zero_padding_a)
            num_non_zero_elems_a = num_non_zero_elems_a - zero_padding_a
        else:
            self.zero_padding = np.empty(shape=(0,))

        # deal with the case of a very small analysis window
        #
        if self.frame_size < num_non_zero_elems_a:
            self.small_window_flag = True
            num_zeros_for_small_window = self.calc_size - self.frame_size
            self.zero_padding = np.zeros(num_zeros_for_small_window)
        else:
            self.small_window_flag = False

        # establish distance between windows (slice size)
        # must be at an integer >= 1
        #
        self.stride_size = int(time_scale_a / self.ratio_t_scale_to_slice_dist)
        self.stride_size = max(1, self.stride_size)

        # determine the indices used to set the analysis window here we look
        # both backward and forward in time earch for half num_non_zero_elems_a.
        # The addition of the % 2 to the r_bound computation allows odd numbers
        #
        left_indices = range(-num_non_zero_elems_a / 2,
                             self.frame_size + 1 - (num_non_zero_elems_a / 2),
                             self.stride_size)

        right_indices = range(num_non_zero_elems_a / 2,
                              self.frame_size + 1 + (num_non_zero_elems_a / 2),
                              self.stride_size)
        self.index_pairs = zip(left_indices, right_indices)

    def make_window_array_for_small_frame(self):
        taper_array = self.taper.taper_fctn(len(self.frame))
        win = self.frame * taper_array

        self.stride_size = 1
        return [np.concatenate((win, self.zero_padding), axis=0) \
                for i in range(0,
                               self.frame_size + 1,
                               self.stride_size)]

    # method: make_window_array
    #
    # arguments: None
    #
    # return:
    #  - window_list: array of windows
    #
    # This does the work of creating an array of data windows with
    # with zero-padding if appropriate, and tapering (aka enveloping,
    # aka windowing) if appropriate. These are used to compute allow
    # various dsp calculations such as rms (energy plot), or fft
    # (frame plot)
    # 
    # For example, in the frame usage case, this method should (probably)
    # only be called as one of the first steps in
    # DemoSpectrogram.do_plot() or DemoEnergyPlot.do_plot(), etc.
    #
    def make_window_array(self):

        # deal with the exeptional case of very very  small time scales
        # we have to call self.taper.taper_fctn directly here,
        # rather than use the prestored self.taper.taper_array
        #
        if self.small_window_flag:
            return self.make_window_array_for_small_frame()


        # initialize window_list to empty list
        #
        window_list = np.empty([len(self.index_pairs), self.calc_size])

        beginning_of_buffer_list = list(filter(
            lambda index_pair: index_pair[0] < 0,
            self.index_pairs))
        size_beg_list = len(beginning_of_buffer_list)

        middle_of_buffer_list = list(filter(
            lambda index_pair: (index_pair[0] >= 0
                           and index_pair[1] <= self.frame_size),
            self.index_pairs))
        size_mid_list = len(middle_of_buffer_list)

        end_of_buffer_list = list(filter(
            lambda index_pair: index_pair[1] > self.frame_size,
            self.index_pairs))

        window_list[0:size_beg_list] = \
            map(self.deal_with_window_beginning_of_frame,
                beginning_of_buffer_list)

        window_list[size_beg_list:size_beg_list + size_mid_list] = \
            map(self.deal_with_window_middle_of_frame,
                middle_of_buffer_list)

        # in the case of very small window length, we must skip this.
        # the end of buffer list will be empty
        #
        try:
            window_list[size_beg_list + size_mid_list:len(window_list)] = \
                map(self.deal_with_window_end_of_frame,
                    end_of_buffer_list)
        except:
            pass

        return window_list
    #
    # end of method

    def deal_with_window_middle_of_frame(self,
                                         bounds):
        l_bound = bounds[0]
        r_bound = bounds[1]
        win = self.frame[l_bound:r_bound] 

        if self.use_taper:
            win = win * self.taper.taper_array

        return np.concatenate((win, self.zero_padding), axis=0)

    def deal_with_window_beginning_of_frame(self,
                                            bounds):
        l_bound = bounds[0]
        r_bound = bounds[1]
        zero_fill = np.zeros(-l_bound)
        win = self.frame[0:r_bound]
        if self.use_taper:
            win = win * self.taper.taper_fctn(r_bound)
        return np.concatenate((win, zero_fill, self.zero_padding), axis=0)

    def deal_with_window_end_of_frame(self,
                                      bounds):
        l_bound = bounds[0]
        r_bound = bounds[1]
        zero_fill = np.zeros(r_bound - self.frame_size)
        win = self.frame[l_bound:self.frame_size]
        if self.use_taper:
            win = win * self.taper.taper_fctn(len(win))
        return np.concatenate((win, zero_fill, self.zero_padding), axis=0)

        

    def square_all_points(self):
        self.frame = map(lambda data_point: data_point ** 2, self.frame)
