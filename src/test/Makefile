CC = gcc
WFLAGS = -Wall -Werror
LDLIBS = -lm -I ../lib
TESTS = not xor

all: tests

%: %.c
	$(CC) $@.c $(WFLAGS) $(LDLIBS) -o $@

tests: $(TESTS)

clean:
	rm -r $(TESTS)
