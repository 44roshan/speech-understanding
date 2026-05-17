def next_birthday(date, birthdays):
    '''
    Find the next birthday after the given date.

    @param:
    date - tuple (month, day)

    birthdays - dictionary:
        {
            (month, day): [names]
        }

    @return:
    birthday - next birthday date
    list_of_names - names having birthday on that date
    '''

    current_month, current_day = date

    # Convert current date to comparable number
    current_value = current_month * 100 + current_day

    next_date = None
    smallest_difference = 10000

    # Check every birthday date
    for bday in birthdays:

        month, day = bday

        # Convert birthday to comparable number
        birthday_value = month * 100 + day

        # If birthday already passed this year
        if birthday_value <= current_value:
            difference = (1200 - current_value) + birthday_value
        else:
            difference = birthday_value - current_value

        # Find smallest future difference
        if difference < smallest_difference:
            smallest_difference = difference
            next_date = bday

    # Get names for that birthday
    list_of_names = birthdays[next_date]

    return next_date, list_of_names


# Example
birthdays = {
    (1, 10): ['Harry', 'Thomas'],
    (3, 20): ['Bobby'],
    (5, 5): ['Charlie'],
    (12, 25): ['David']
}

result = next_birthday((3, 15), birthdays)

print(result)
